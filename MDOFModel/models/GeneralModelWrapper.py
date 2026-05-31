import openseespy.opensees as ops
import numpy as np
from pathlib import Path
from typing import Callable, List, Optional
import opsvis as opsv
import matplotlib.pyplot as plt
import sys
import eqsig.single

from ..analysis.ReadRecord import ReadRecord

class GeneralModelWrapper:
    """
    一个用于将通用二维或三维 OpenSeesPy 模型封装以供 MDOFModel.IDA 并行分析模块使用的适配器。
    只要你通过该 Wrapper 指定哪一层对应哪些节点、每层的高度，IDA 模块就可以正常提取位移、漂移和加速度等 EDP 并在多进程中反复调用。

    当前 IDA -> Pelicun 损失评估工作流默认采用 mm-N-s 单位体系：
    - 节点坐标、层高和位移单位为 mm；
    - 加速度单位为 mm/s²；
    - 速度单位为 mm/s；
    - 地震波输入记录通常为 g，默认通过 ``g_factor=9800.0`` 转换到 mm/s²。
    """

    # === 公开属性 (供 IDA 或外部提取的结果与配置) ===
    UniqueRecorderPrefix: str = "General_"
    """str: 多进程并发分析时的记录器输出文件前缀，用于避免不同进程间产生文件读写冲突。"""

    TmpDir: Path = Path(".opensees_tmp")
    """pathlib.Path: OpenSees Recorder 及日志等临时文件的输出根目录。默认为工作目录下的 .opensees_tmp/，不会被 git 追踪。可在实例化后修改。"""
    
    MaxDrift: list = []
    """list[float]: 单次动力分析完成后提取的各楼层最大层间位移角。格式为浮点数列表，长度对应总楼层数量（如 [一层角, 二层角, ...]）。"""
    
    MaxAbsAccel: list = []
    """list[float]: 单次动力分析完成后提取的各层最大绝对加速度（默认单位 mm/s²）。
    若提供了 base_nodes，列表长度 = len(floor_nodes) + 1，index 0 为地面 PGA，其余为各楼层绝对加速度；
    未提供 base_nodes 时长度 = len(floor_nodes)，不含地面值。
    """
    
    MaxRelativeAccel: list = []
    """list[float]: 单次动力分析完成后提取的各层最大相对加速度（默认单位 mm/s²，相对地面的加速度）。
    若提供了 base_nodes，列表长度 = len(floor_nodes) + 1，index 0 为地面节点相对加速度（恒为 0），其余为各楼层相对加速度；
    未提供 base_nodes 时长度 = len(floor_nodes)，不含地面值。
    """    

    MaxAbsVel: list = []
    """list[float]: 单次动力分析完成后提取的各层最大绝对速度（默认单位 mm/s）。
    若提供了 base_nodes，列表长度 = len(floor_nodes) + 1，index 0 为地面 PGV，其余为各楼层绝对速度；
    未提供 base_nodes 时长度 = len(floor_nodes)，不含地面值。
    """    

    MaxRelativeVel: list = []
    """list[float]: 单次动力分析完成后提取的各层最大相对速度（默认单位 mm/s，相对地面的速度）。
    若提供了 base_nodes，列表长度 = len(floor_nodes) + 1，index 0 为地面节点相对速度（恒为 0），其余为各楼层相对速度；
    未提供 base_nodes 时长度 = len(floor_nodes)，不含地面值。
    """

    ResDrift: float = 0.0
    """float: 单次动力分析结束后整个结构（通常为全楼层中出现的最大值）的残余层间位移角。"""

    TotalWeight: float = 0.0
    """float: 自动重力分析中根据节点质量和 g_factor 计算得到的结构总重力。"""
    
    DriftHistory: dict = {}
    """dict[int|str, numpy.ndarray]: 静力推覆(Pushover)等分析后记录的各层位移角时程。包含键 'time' 以及楼层号 (1, 2, ...)，对应值为该层随全过程的一维浮点数 numpy 数组。"""
    
    NodeDispHistory: dict = {}
    """dict[int|str, numpy.ndarray]: 静力推覆(Pushover)等分析后记录的控制节点位移时程。结构同样包含键 'time' 及各层楼层号，对应值为位移过程的浮点数组。"""
    
    DampingRatio: float = 0.05
    """float: 模型全局阻尼比。默认 0.05。在自动调用模态分析得到前两阶频率后，借由此项计算两振型瑞利(Rayleigh)阻尼参数。"""
    
    # === 可调用的回调函数 (公开Callable) ===
    build_model_func: Callable
    """Callable: 建立有限元基础模型的回调函数。仅包含定义节点、材料、截面、单元等步骤，不能包含 ops.wipe()。"""

    # === 模型配置内置参数 (私有) ===
    _floor_nodes: List[int]
    """list[int]: [内部参数] 竖向每一层结构作为记录提取目标的控制节点标签列表 (如底向上：[103, 203, ...])。"""
    
    _story_heights: List[float]
    """list[float]: [内部参数] 各连续两层之间的层高列表。用于自动换算节点位移至层间位移角。"""
    
    _base_nodes: List[int]
    """list[int]: [内部参数] 结构底部的节点标签列表（通常为嵌固端），用于相对相对位移和绝对加速度运算的基准参照。"""
    
    _dof: int
    """int: [内部参数] 主要控制、提取及地震激励所作用的自由度方向编号 (例如 1代表X向，2代表Y向，3代表Z向)。"""
    
    _g_factor: float
    """float: [内部参数] 重力加速度及地震波加速度比例系数。默认 9800.0，对应 mm-N-s 单位体系。"""

    T1: float = 0.0
    """float: 结构在指定自由度方向的第一阶基本平动周期，初始化时自动计算。当无法完成特征值屈曲分析时为 0.0。"""

    extra_recorder_setup: Optional[Callable]
    """
    Optional[Callable[[Path], None]]: 用户自定义 recorder 的注册回调。

    在每次 :meth:`DynamicAnalysis` 的标准 recorder 创建完成后调用，参数为当次分析的
    唯一临时目录 ``tmp_dir: Path``。在此回调内可使用任意 ``ops.recorder(...)`` 命令，
    输出文件应放入 ``tmp_dir`` 以保证多进程安全。
    若为 ``None`` （默认）则不执行任何额外操作。

    示例::

        def setup(tmp_dir):
            ops.recorder('EnvelopeElement', '-file', str(tmp_dir / 'strain.out'), '-ele', 101, 'section', 1, 'fiber', 0, 0, 'strain')
        wrapper.extra_recorder_setup = setup
    """

    extra_post_process: Optional[Callable]
    """
    Optional[Callable[[GeneralModelWrapper, Path], None]]: 自定义 EDP 后处理回调。
    
    在 ``ops.wipe()`` 和标准 :meth:`_post_process` 完成后调用，参数为 ``(self, tmp_dir)``。
    在此回调内读取自定义 recorder 输出文件并将结果写入 ``self.xxx`` 属性，
    之后即可通过 ``ExtraEDP={'EDP名': 'xxx'}`` 被 IDA 模块读取。
    若为 ``None`` （默认）则不执行任何额外操作。

    示例::

        def post(model, tmp_dir):
            import numpy as np
            data = np.loadtxt(tmp_dir / 'strain.out')
            model.MaxColStrain = [float(np.atleast_2d(data)[2, 0])]
        wrapper.extra_post_process = post
    """

    def __init__(self, build_model_func: Callable, floor_nodes: List[int], story_heights: List[float], dof: int = 1, base_nodes: Optional[List[int]] = None, g_factor: float = 9800.0):
        """
        Parameters
        ----------
        build_model_func : Callable
            建立基本模型的无参函数（例如封装了节点、单元、截面创建的函数）。内部不应包含 ops.wipe()，如果需要，可以在外部包装。
        floor_nodes : List[int]
            每层位移/加速度记录所对应的目标节点标签列表（从下到上，例如 [103, 203, 303...]）。
        story_heights : List[float]
            每层的层高（对应 floor_nodes 的数量，必须相同）。默认单位为 mm；例如第一层高 3m、第二层高 3m，则为 [3000, 3000]。
        dof : int, optional
            地震输入与反应提取的自由度方向，默认 1 (x方向)。
        base_nodes : List[int], optional
            底层或基底节点标签列表。用于计算第一层相对基底的位移与漂移。如果不提供，默认为 [1] 或者认为基底无位移。
        g_factor : float, optional
            将 'g' 单位转换为模型物理单位的倍乘系数。默认 9800.0，对应 mm/s²。
            目前 IDA -> Pelicun 后处理假定 GeneralModelWrapper 的 IDA 输出采用 mm/s² 和 mm/s。
        """
        self.build_model_func = build_model_func
        self._floor_nodes = floor_nodes
        self._story_heights = story_heights
        self._dof = dof
        self._base_nodes = base_nodes if base_nodes else []
        self._g_factor = g_factor
        self.DampingRatio = 0.05
        self.TmpDir = Path(".opensees_tmp")

        # 满足 IDA 并行调用需要的 UniqueRecorderPrefix
        self.UniqueRecorderPrefix = "General_"
        
        # IDA 规定必须存在的 4 个 EDP 存储属性
        self.MaxDrift = []
        self.MaxAbsAccel = []
        self.MaxRelativeAccel = []
        self.MaxAbsVel = []
        self.MaxRelativeVel = []
        self.ResDrift = []
        self.TotalWeight = 0.0

        # 用户自定义 EDP 回调（默认不启用）
        self.extra_recorder_setup = None
        self.extra_post_process   = None

        # 自动计算基本周期
        ops.wipe()
        self.build_model_func()
        try:
            _, periods = self.ModalAnalysis(num_modes=2, ifprint=False)
            if not periods:
                raise ValueError("ModalAnalysis returned empty periods.")
            self.T1 = periods[0]
        except Exception as e:
            ops.wipe()
            raise RuntimeError(f"自动计算基本周期 (ModalAnalysis) 失败。请检查模型配置、质量矩阵定义或通过参数手动指定周期。错误信息: {str(e)}")
        ops.wipe()

    def _auto_apply_gravity(self):
        """自动根据节点的平动质量在重力方向(Y或Z)施加向下的重力荷载并完成静力分析"""
        ops.timeSeries('Linear', 100)
        ops.pattern('Plain', 101, 100)
        self.TotalWeight = 0.0
        
        node_tags = ops.getNodeTags()
        if not node_tags:
            return
            
        ndm = len(ops.nodeCoord(node_tags[0]))
        for tag in node_tags:
            masses = ops.nodeMass(tag)
            if masses:
                # 获取各个平动方向的最大质量，假设为此节点的物理质量
                trans_mass = max(masses[:ndm])
                if trans_mass > 1e-6:
                    weight_force = -trans_mass * self._g_factor
                    self.TotalWeight += abs(weight_force)
                    if ndm == 2:
                        ops.load(tag, 0.0, weight_force, 0.0)
                    elif ndm == 3:
                        ops.load(tag, 0.0, 0.0, weight_force, 0.0, 0.0, 0.0)
                        
        ops.constraints('Transformation')
        ops.numberer('RCM')
        ops.system('BandGeneral')
        ops.test('NormDispIncr', 1.0e-6, 50)
        ops.algorithm('Newton')
        ops.integrator('LoadControl', 0.1)
        ops.analysis('Static')
        
        ok = ops.analyze(10)
        if ok != 0:
            ops.algorithm('KrylovNewton')
            ok = ops.analyze(10)
            
        if ok != 0:
            print('Warning: Auto gravity analysis failed to converge.')
            
        ops.loadConst('-time', 0.0)
        ops.wipeAnalysis()

    def DynamicAnalysis(self, record_file: str, scale_factor: float, ifprint: bool = False, delta_t='AsInRecord', animate: bool = False, show_progress: bool = False, **kwargs):
        """
        供 IDA.__init__ / IDA_1record 内部无感调用的核心方法：包含清空、重塑模型、阻尼、地震动激励及求解全流程。

        Parameters
        ----------
        record_file : str
            地震动记录文件路径。支持 PEER NGA 强震数据库标准格式（.AT2）或无表头的等时间步长单列加速度数据文本文件；记录数据的单位通常应为重力加速度 g，程序内部将根据初始化时指定的 g_factor 自动换算为模型对应的物理单位。
        scale_factor : float
            地震动记录的缩放因子。
        ifprint : bool
            是否打印分析过程信息。
        delta_t : str or float
            时间步长。
        animate : bool
            是否在分析结束后自动播放 openseespy 动态位移变形动画。默认 False。
        show_progress : bool
            是否在此方法内的隐式时间积分循环中打印控制台进度条信息，建议单次分析排错或观测性能时可开启，如果上层使用多进程并发分析建议关掉以防止控制台刷屏。默认 False。
        kwargs : dict
            传递给 opsvis.anim_defo() 的其他绘图参数，例如 xlim, ylim, sfac, skip_steps 等。

        Returns
        -------
        Tuple[bool, float, float]
            分析是否成功、当前时间、总时间。
        """
        # 读取并设置地震波文件
        p = Path(record_file)
        self.UniqueRecorderPrefix = p.stem
        prefix = self.UniqueRecorderPrefix

        # 所有临时文件（含日志）统一放入 TmpDir/<prefix>，不自动清理
        _tmp_dir = self.TmpDir / f"opensees_{prefix}"
        _tmp_dir.mkdir(parents=True, exist_ok=True)
        log_file = _tmp_dir / "opensees.log"
        temp_eq_file = _tmp_dir / f"temp_{p.name}.dat"

        ops.wipe()
        ops.logFile(log_file.as_posix(), "-noEcho")

        # 1. 建立模型和重力
        self.build_model_func()
        self._auto_apply_gravity()

        # 2. 自动施加阻尼 (使用内置模态计算与Rayleigh阻尼)
        omegas, _ = self.ModalAnalysis(num_modes=5, ifprint=ifprint)
        if len(omegas) >= 2:
            alpha_m = self.DampingRatio * (2.0 * omegas[0] * omegas[1]) / (omegas[0] + omegas[1])
            beta_k_init = 2.0 * self.DampingRatio / (omegas[0] + omegas[1])
            ops.rayleigh(alpha_m, 0.0, beta_k_init, 0.0)
        dt_gm, nPts = ReadRecord(record_file, temp_eq_file.as_posix())

        # 预计算地面速度时程（积分地面加速度），供绝对速度 Recorder 的 -timeSeries 引用
        _eq_accel_arr = np.array(open(temp_eq_file).read().split(), dtype=float) * (scale_factor * self._g_factor)
        _eq_vel_arr   = np.cumsum(_eq_accel_arr) * dt_gm
        vel_ts_file   = _tmp_dir / "ground_vel.dat"
        np.savetxt(vel_ts_file, _eq_vel_arr, fmt='%.9e')

        ana_dt = dt_gm if delta_t == 'AsInRecord' else float(delta_t)

        # 4. 定义地震动激励及速度时程（需在 Recorder 引用前定义）
        ops.timeSeries("Path", 111, "-dt", dt_gm, "-filePath", temp_eq_file.as_posix(), "-factor", scale_factor * self._g_factor)
        ops.timeSeries("Path", 112, "-dt", dt_gm, "-filePath", vel_ts_file.as_posix(), "-factor", 1.0)
        ops.pattern("UniformExcitation", 111, self._dof, "-accel", 111)

        disp_file          = _tmp_dir / "disp.out"
        abs_accel_env_file = _tmp_dir / "abs_accel_env.out"
        rel_accel_env_file = _tmp_dir / "rel_accel_env.out"
        abs_vel_env_file   = _tmp_dir / "abs_vel_env.out"
        rel_vel_env_file   = _tmp_dir / "rel_vel_env.out"

        # 加速度/速度 Recorder 节点列表：index 0 为地面节点（取 base_nodes[0]，若有），其余为各楼层节点
        # 地面节点（固定，相对加速度/速度 = 0）：
        #   accel + timeSeries 111 → PGA；vel + timeSeries 112 → PGV
        # 楼层节点（index 1+）：
        #   accel + timeSeries 111 → 绝对加速度；vel + timeSeries 112 → 绝对速度
        #   不加 timeSeries → 相对加速度/速度（相对地面）
        _gnd_node = self._base_nodes[0] if self._base_nodes else None
        _acc_vel_nodes = ([_gnd_node] + self._floor_nodes) if _gnd_node is not None else self._floor_nodes

        # 位移时程（供层间漂移和残余漂移后处理）
        ops.recorder("Node", "-file", disp_file.as_posix(), "-time", "-node", *self._floor_nodes, "-dof", self._dof, "disp")
        # EnvelopeNode 直接输出 min/max/absMax 三行，避免保存完整加速度/速度时程
        ops.recorder("EnvelopeNode", "-file", abs_accel_env_file.as_posix(), "-timeSeries", 111, "-node", *_acc_vel_nodes, "-dof", self._dof, "accel")
        ops.recorder("EnvelopeNode", "-file", rel_accel_env_file.as_posix(),                     "-node", *_acc_vel_nodes, "-dof", self._dof, "accel")
        ops.recorder("EnvelopeNode", "-file", abs_vel_env_file.as_posix(),   "-timeSeries", 112, "-node", *_acc_vel_nodes, "-dof", self._dof, "vel")
        ops.recorder("EnvelopeNode", "-file", rel_vel_env_file.as_posix(),                       "-node", *_acc_vel_nodes, "-dof", self._dof, "vel")

        base_disp_file = None
        if self._base_nodes:
            base_disp_file = _tmp_dir / "basedisp.out"
            ops.recorder("Node", "-file", base_disp_file.as_posix(), "-time", "-node", *self._base_nodes, "-dof", self._dof, "disp")

        # 用户自定义 recorder（在标准 recorder 之后、分析开始之前注册）
        if self.extra_recorder_setup is not None:
            self.extra_recorder_setup(_tmp_dir)
        
        # 5. 瞬态分析设置 (可根据需要替换)
        ops.system("BandGeneral")
        ops.numberer("RCM")
        ops.constraints("Transformation")
        ops.integrator("Newmark", 0.5, 0.25)
        ops.test('NormDispIncr', 1.0e-3, 100, 0)
        ops.algorithm("Newton")
        ops.analysis("VariableTransient", "-numSubLevels", 4, "-numSubSteps", 2)
        
        n_steps = int((nPts * dt_gm) / ana_dt)
        finished = True
        
        # 动画设置：如果 animate=True，则设置xlim和ylim以适应模型尺寸，记录每步的位移数据供后续动画使用
        if animate:
            anim_ele_tags = ops.getEleTags()
            Eds_list = []
            time_list = []
            if 'xlim' not in kwargs or 'ylim' not in kwargs:
                node_tags = ops.getNodeTags()
                if node_tags:
                    coords = np.array([ops.nodeCoord(tag) for tag in node_tags])
                    if coords.size > 0:
                        xmin, xmax = np.min(coords[:, 0]), np.max(coords[:, 0])
                        ymin, ymax = np.min(coords[:, 1]), np.max(coords[:, 1])
                        dx = max(xmax - xmin, 1.0)
                        dy = max(ymax - ymin, 1.0)
                        if 'xlim' not in kwargs:
                            kwargs['xlim'] = [xmin - dx * 0.5, xmax + dx * 0.5]
                        if 'ylim' not in kwargs:
                            kwargs['ylim'] = [ymin - dy * 0.1, ymax + dy * 0.1]
        
        # 定义一个简单的进度输出间隔以避免刷屏，例如每 5% 刷新一次控制台打印
        progress_interval = max(1, n_steps // 20)

        for step in range(n_steps):
            
            if show_progress and (step % progress_interval == 0 or step == n_steps - 1):
                percent = (step + 1) / n_steps * 100
                sys.stdout.write(f"\r[{self.UniqueRecorderPrefix}] Dynamic Analysis: [{int(percent / 5) * '#'}{'.' * (20 - int(percent / 5))}] {percent:.1f}% (Step {step+1}/{n_steps})")
                sys.stdout.flush()

            ok = ops.analyze(1, ana_dt)

            if ok != 0:
                # Fallback
                ops.algorithm('KrylovNewton')
                ops.test('NormDispIncr', 1.0e-2, 100, 0)
                ok = ops.analyze(1, ana_dt)
                if ok != 0:
                    ops.algorithm('ModifiedNewton')
                    ok = ops.analyze(1, ana_dt)
                ops.test('NormDispIncr', 1.0e-3, 100, 0)
                ops.algorithm('Newton')
                
            if ok != 0:
                finished = False
                break
                
            anim_step_interval = max(1, int(0.1 / ana_dt))  # 控制帧数为 10 帧/秒，即每 0.1 秒记录一次
            
            # 记录动画数据：根据模拟时长控制帧数，每 0.1 秒获取并存储一次（即 10 帧/秒），并保存最后一步
            if animate and (step % anim_step_interval == 0 or step == n_steps - 1):
                time_list.append(ops.getTime())
                ed = []
                for ele_tag in anim_ele_tags:
                    nodes = ops.eleNodes(ele_tag)
                    if len(nodes) >= 2:
                        try:
                            d1 = ops.nodeDisp(nodes[0])
                            d2 = ops.nodeDisp(nodes[1])
                            d1 = d1 + [0.0]*(3-len(d1)) if len(d1)<3 else d1[:3]
                            d2 = d2 + [0.0]*(3-len(d2)) if len(d2)<3 else d2[:3]
                            ed.append(d1 + d2)
                        except:
                            ed.append([0.0]*6)
                    else:
                        ed.append([0.0]*6)
                Eds_list.append(ed)
                
        if show_progress:
            print() # 换行收尾
            
        tCurrent = ops.getTime()
        totalTime = n_steps * ana_dt
        
        # 动画展示：仅在分析完成后进行，避免对求解过程性能产生影响
        if animate and len(Eds_list) > 0:
            Eds_arr = np.array(Eds_list)
            time_arr = np.array(time_list)
            
            skip_steps = kwargs.pop('skip_steps', 1)
            sfac = kwargs.pop('sfac', 10.0) # opsvis需要一个合理缩放引子
            
            if skip_steps > 1:
                Eds_arr = Eds_arr[::skip_steps]
                time_arr = time_arr[::skip_steps]
            
            if 'fmt_defo' not in kwargs:
                kwargs['fmt_defo'] = {'color': 'blue', 'linestyle': 'solid', 'linewidth': 2.0, 'marker': '', 'markersize': 1}
            
            print(f'Starting to animate deformed shape with opsvis. {len(time_arr)} frames...')
            
            try:
                anim = opsv.anim_defo(Eds_arr, time_arr, sfac, **kwargs)
                plt.show()
                # Ensure object is kept alive during show
                self._current_anim_obj = anim
            except Exception as e:
                print(f"Warning: Failed to render animation: {e}")
                
        # Wipe 会自动 flush file recorder，以确保可以读取
        ops.wipe()
        
        # 6. 后处理：从包络文件读取 EDP 最大值，从位移时程中计算漂移和残余漂移
        self._post_process(disp_file, abs_accel_env_file, rel_accel_env_file, abs_vel_env_file, rel_vel_env_file, base_disp_file)

        # 用户自定义 EDP 后处理（读取自定义 recorder 文件并设置属性）
        if self.extra_post_process is not None:
            self.extra_post_process(self, _tmp_dir)
            
        return finished, tCurrent, totalTime

    def DynamicAnalysis_Sa(self, record_file: str, target_Sa: float, ifprint: bool = False, delta_t='AsInRecord', animate: bool = False, show_progress: bool = False, **kwargs):
        """
        以目标谱加速度 Sa(T₁, ζ)（单位 g）为输入做动力时程分析。
        内部用 eqsig 计算原始记录在结构基本周期 T₁ 处的谱加速度，
        进而推算所需缩放系数后调用 :meth:`DynamicAnalysis`。
        响应谱计算使用 ``self.DampingRatio`` 作为阻尼比。

        Parameters
        ----------
        record_file : str
            地震动记录文件路径（与 :meth:`DynamicAnalysis` 相同）。
        target_Sa : float
            目标谱加速度，单位 g。
        ifprint : bool
            是否打印分析过程信息。
        delta_t : str or float
            时间步长。
        animate : bool
            是否在分析结束后自动播放 openseespy 动态位移变形动画。默认 False。
        show_progress : bool
            是否打印控制台进度条信息。默认 False。
        **kwargs
            其余参数透传给 opsvis.anim_defo()。

        Returns
        -------
        Tuple[bool, float, float]
            与 :meth:`DynamicAnalysis` 返回值相同：
            (分析是否成功, 当前时间, 总时间)。

        Raises
        ------
        ValueError
            若 T₁ ≤ 0 或记录的 Sa(T₁) ≈ 0，无法计算缩放系数。
        """

        if self.T1 <= 0.0:
            raise ValueError(
                f"结构基本周期 T1={self.T1:.4f} s 无效，无法计算 Sa(T1)。"
                "请检查模型质量矩阵或手动设置 wrapper_model.T1。"
            )

        # ── 读取地震动，获取 dt 及加速度序列（g） ──────────────────────────
        p = Path(record_file)
        _sa_tmp_dir = self.TmpDir / "sa_calc"
        _sa_tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = (_sa_tmp_dir / ('temp_' + p.name + '.dat')).as_posix()
        dt_gm, _ = ReadRecord(record_file, tmp_path)
        if dt_gm is None:
            raise FileNotFoundError(f"无法读取地震动记录: {record_file}")
        with open(tmp_path, 'r') as f:
            accel_g = np.array(f.read().split(), dtype=float)

        # ── 用 eqsig 计算原始记录在 T1 处的 Sa（g） ─────────────────────
        record = eqsig.single.AccSignal(accel_g * 9.8, dt_gm)
        record.generate_response_spectrum(response_times=np.array([self.T1]), xi=self.DampingRatio)
        Sa_record = record.s_a[0] / 9.8

        if Sa_record < 1.0e-10:
            raise ValueError(
                f"记录在 T1={self.T1:.3f}s 处的 Sa 接近于零 ({Sa_record:.3e} g)，"
                "无法确定缩放系数。请检查记录文件或周期设置。"
            )

        scale_factor = target_Sa / Sa_record

        print(f"  T1 = {self.T1:.3f} s | "
              f"Sa(T1, record) = {Sa_record:.4f} g | "
              f"target Sa = {target_Sa:.4f} g | "
              f"scale_factor = {scale_factor:.4f}")

        return self.DynamicAnalysis(
            record_file=record_file,
            scale_factor=scale_factor,
            ifprint=ifprint,
            delta_t=delta_t,
            animate=animate,
            show_progress=show_progress,
            **kwargs,
        )

    def _post_process(self, disp_file, abs_accel_env_file, rel_accel_env_file, abs_vel_env_file, rel_vel_env_file, base_disp_file):
        """从 EnvelopeNode 文件直接读取加速度/速度最大值；从位移时程中计算最大层间漂移和残余漂移。
        
        加速度/速度数组长度始终为 n+1（地面节点 index 0 + n 个楼层），
        其中 index 0 为地面值（PGA / PGV / 0）。
        """
        n = len(self._floor_nodes)
        n_out = n + 1
        zeros = [0.0] * n_out

        def _read_env_absmax(path):
            """读取 EnvelopeNode 输出文件的 absMax 行（第3行），返回浮点列表；失败则返回 None。"""
            try:
                data = np.loadtxt(path)
                arr  = np.atleast_2d(data)  # 统一转为 2D
                if arr.shape[0] == 3:       # 标准格式：3行(min/max/absMax) × n_nodes列
                    return arr[2, :].tolist()
            except Exception:
                pass
            return None

        abs_accel = _read_env_absmax(abs_accel_env_file)
        rel_accel = _read_env_absmax(rel_accel_env_file)
        abs_vel   = _read_env_absmax(abs_vel_env_file)
        rel_vel   = _read_env_absmax(rel_vel_env_file)

        self.MaxAbsAccel      = abs_accel if abs_accel is not None else zeros[:]
        self.MaxRelativeAccel = rel_accel if rel_accel is not None else zeros[:]
        self.MaxAbsVel        = abs_vel   if abs_vel   is not None else [1e-6] * n_out
        self.MaxRelativeVel   = rel_vel   if rel_vel   is not None else [1e-6] * n_out

        # ── 位移时程：最大层间漂移 + 残余漂移 ──────────────────────────────
        try:
            disp_data = np.loadtxt(disp_file)
        except Exception:
            self.MaxDrift = [0.0] * n
            self.ResDrift = 0.0
            return

        if disp_data.size == 0:
            self.MaxDrift = [0.0] * n
            self.ResDrift = 0.0
            return

        if disp_data.ndim == 1:
            disp_data = disp_data.reshape(1, -1)

        Times = disp_data[:, 0]
        Disps = disp_data[:, 1:]

        BaseDisps = np.zeros(len(Times))
        if base_disp_file and base_disp_file.exists():
            try:
                base_data = np.loadtxt(base_disp_file)
                if base_data.ndim == 1:
                    base_data = base_data.reshape(1, -1)
                if base_data.shape[1] > 1:
                    BaseDisps = np.mean(base_data[:, 1:], axis=1)
            except Exception:
                pass

        num_floors  = Disps.shape[1]
        num_steps   = Disps.shape[0]
        FloorDrifts = np.zeros((num_steps, num_floors))

        for i in range(num_floors):
            lower_disp = BaseDisps if i == 0 else Disps[:, i - 1]
            FloorDrifts[:, i] = (Disps[:, i] - lower_disp) / self._story_heights[i]

        self.MaxDrift = np.max(np.abs(FloorDrifts), axis=0).tolist()

        # 残余漂移：取最后几秒漂移时程的均值，减少末尾振动噪声的影响
        if len(Times) > 1:
            last_seconds = max(5.0, 0.1 * float(Times[-1] - Times[0]))
            mask = Times >= (float(Times[-1]) - last_seconds)
        else:
            mask = np.ones(len(Times), dtype=bool)
        self.ResDrift = float(np.max(np.abs(np.mean(FloorDrifts[mask, :], axis=0)))) if np.any(mask) else 0.0

    def StaticPushover(self, maxU: List[float] = [0.10, -0.10, 0.0], dU: float = 0.001, CFloor='roof', ifprint: bool = True, lateral_load_pattern_func: Optional[Callable] = None, animate: bool = False, **kwargs):
        """
        静力推覆分析 (Static Pushover Analysis)。
        
        Parameters
        ----------
        maxU : List[float], optional
            目标节点推覆的位移目标列表 (默认 [0.10, -0.10, 0.0])。
        dU : float, optional
            位移控制的增量步长。
        CFloor : str or int, optional
            推覆控制层。'roof' 代表顶层，或者传入层号如 1, 2。
        ifprint : bool, optional
            是否在分析期间打印提示。
        lateral_load_pattern_func : Callable, optional
            如果提供了此函数，将用于施加推覆荷载。若是 None，将采用与楼层高度成正比的倒三角模式加载。
        animate : bool, optional
            是否在分析结束后自动播放推覆动画，同时绘制推覆曲线（基底剪力系数 V/W - 顶点位移角）。
        kwargs : dict
            传递给 opsvis.anim_defo() 的其他绘图参数。
            
        Returns
        -------
        Tuple[bool, float]
            分析是否失败标识 Iffinish（失败为 True，成功为 False），以及当前节点位移。
        """
        if ifprint:
            print('Pushover analysis via GeneralModelWrapper with OpenSeesPy...')
            
        ops.wipe()
        
        # 1. 建立模型和施加重力
        self.build_model_func()
        self._auto_apply_gravity()

        # 2. 从参数中确定控制节点
        if isinstance(CFloor, str) and (CFloor == 'roof'):
            ctrl_node = self._floor_nodes[-1]
        elif isinstance(CFloor, int):
            ctrl_node = self._floor_nodes[CFloor - 1]
        else:
            ctrl_node = self._floor_nodes[-1]

        # 3. 施加推覆荷载 (时程Tag、Pattern Tag可自定)
        tsTag = 301
        ops.timeSeries('Linear', tsTag)
        patternTag = 201
        ops.pattern('Plain', patternTag, tsTag)

        if lateral_load_pattern_func is not None:
            lateral_load_pattern_func()
        else:
            # 默认：倒三角模式，按各层相对底部的累积高度分布水平推力
            total_height = sum(self._story_heights)
            cum_h = 0.0
            for i, node_tag in enumerate(self._floor_nodes):
                cum_h += self._story_heights[i]
                # 获取节点自由度数量
                ndf = len(ops.nodeDisp(node_tag))
                val = [0.0] * ndf
                val[self._dof - 1] = cum_h / total_height
                ops.load(node_tag, *val)
        
        # 4. 设置 Recorder
        prefix = self.UniqueRecorderPrefix
        _tmp_dir = self.TmpDir / f"opensees_{prefix}_push"
        _tmp_dir.mkdir(parents=True, exist_ok=True)
        disp_file = _tmp_dir / "push_disp.out"
        base_disp_file = None
        base_reaction_file = None
        
        ops.recorder("Node", "-file", disp_file.as_posix(), "-time", "-node", *self._floor_nodes, "-dof", self._dof, "disp")
        if self._base_nodes:
            base_disp_file = _tmp_dir / "push_basedisp.out"
            ops.recorder("Node", "-file", base_disp_file.as_posix(), "-time", "-node", *self._base_nodes, "-dof", self._dof, "disp")
            
            base_reaction_file = _tmp_dir / "push_reaction.out"
            ops.recorder("Node", "-file", base_reaction_file.as_posix(), "-time", "-node", *self._base_nodes, "-dof", self._dof, "reaction")

        # 5. 初始配置分析
        Tol = 1e-6
        maxNumIter = 100
        
        ops.system('BandGeneral')
        ops.constraints('Transformation')
        ops.numberer('RCM')
        ops.test('NormDispIncr', Tol, maxNumIter)
        ops.algorithm('NewtonLineSearch') 

        currentDisp = 0.0
        ok = 0
        
        # 动画设置
        if animate:
            anim_ele_tags = ops.getEleTags()
            Eds_list = []
            time_list = []
            if 'xlim' not in kwargs or 'ylim' not in kwargs:
                node_tags = ops.getNodeTags()
                if node_tags:
                    coords = np.array([ops.nodeCoord(tag) for tag in node_tags])
                    if coords.size > 0:
                        xmin, xmax = np.min(coords[:, 0]), np.max(coords[:, 0])
                        ymin, ymax = np.min(coords[:, 1]), np.max(coords[:, 1])
                        dx = max(xmax - xmin, 1.0)
                        dy = max(ymax - ymin, 1.0)
                        if 'xlim' not in kwargs:
                            kwargs['xlim'] = [xmin - dx * 0.5, xmax + dx * 0.5]
                        if 'ylim' not in kwargs:
                            kwargs['ylim'] = [ymin - dy * 0.1, ymax + dy * 0.1]

        # 6. 推覆位移循环
        for target in maxU:
            while ok == 0 and abs(currentDisp - target) > dU:
                # 调整步长符号
                ops.integrator('DisplacementControl', ctrl_node, self._dof, np.sign(target - currentDisp) * dU, maxNumIter)
                ops.analysis('Static')
                ok = ops.analyze(1)
                
                if ok != 0:
                    ops.algorithm('ModifiedNewton')
                    ok = ops.analyze(1)
                    if ok != 0:
                        ops.algorithm('KrylovNewton')
                        ok = ops.analyze(1)
                    ops.algorithm('NewtonLineSearch')
                    
                if ok != 0:
                    break
                currentDisp = ops.nodeDisp(ctrl_node, self._dof)
                
                # 记录动画数据：每步获取指定单元的节点位移并存储，供后续动画使用
                if animate:
                    time_list.append(currentDisp)
                    ed = []
                    for ele_tag in anim_ele_tags:
                        nodes = ops.eleNodes(ele_tag)
                        if len(nodes) >= 2:
                            try:
                                d1 = ops.nodeDisp(nodes[0])
                                d2 = ops.nodeDisp(nodes[1])
                                d1 = d1 + [0.0]*(3-len(d1)) if len(d1)<3 else d1[:3]
                                d2 = d2 + [0.0]*(3-len(d2)) if len(d2)<3 else d2[:3]
                                ed.append(d1 + d2)
                            except:
                                ed.append([0.0]*6)
                        else:
                            ed.append([0.0]*6)
                    Eds_list.append(ed)

        Iffinish = not (ok == 0)
        if ifprint:
            state_str = "Failed" if Iffinish else "Successful"
            print(f'Pushover Analysis State: {state_str} (Code: {ok})')
            
        # 停止所有记录器，使得输出流刷入文件，而不销毁模型（使得 opsvis 绘图依然可行）
        ops.remove('recorders')
        
        # 7. 推覆后处理记录读取
        self._post_process_pushover(disp_file, base_disp_file, base_reaction_file)
        
        if animate:
            fig, ax = plt.subplots(figsize=(8, 6))
            total_H = sum(self._story_heights)
            
            # 使用提取到的顶层节点相对基底的位移，计算全局漂移角
            roof_disp = self.NodeDispHistory.get(len(self._floor_nodes), np.zeros(len(self.NodeDispHistory.get('time', []))))
            base_disp = np.zeros(len(roof_disp))
            if base_disp_file and base_disp_file.exists():
                try:
                    base_data = np.loadtxt(base_disp_file)
                    if base_data.ndim == 1: base_data = base_data.reshape(1, -1)
                    if base_data.shape[1] > 1: base_disp = np.mean(base_data[:, 1:], axis=1)
                except:
                    pass
                    
            roof_drift = (roof_disp - base_disp) / total_H if total_H > 0 else roof_disp
            
            shear_coeff = getattr(self, 'BaseShearCoefficientHistory', np.zeros(len(roof_drift)))
            ax.plot(roof_drift, shear_coeff, 'k-', linewidth=2.0)
            ax.set_xlabel('Roof Drift Ratio')
            ax.set_ylabel('Base Shear / Weight')
            ax.set_title('Pushover Capacity Curve')
            ax.grid(True)
            
            if len(Eds_list) > 0:
                print(f'Starting to animate pushover deformed shape with opsvis. {len(time_list)} frames...')
                try:
                    Eds_arr = np.array(Eds_list)
                    time_arr = np.array(time_list)
                    skip_steps = kwargs.pop('skip_steps', 1)
                    sfac = kwargs.pop('sfac', 1.0) # 推覆本身位移较大，通常不需要像动力那样的 sfac=10，默认给 1.0
                    
                    if skip_steps > 1:
                        Eds_arr = Eds_arr[::skip_steps]
                        time_arr = time_arr[::skip_steps]

                    if 'fmt_defo' not in kwargs:
                        kwargs['fmt_defo'] = {'color': 'blue', 'linestyle': 'solid', 'linewidth': 2.0, 'marker': '', 'markersize': 1}

                    anim = opsv.anim_defo(Eds_arr, time_arr, sfac, **kwargs)
                    self._current_anim_obj = anim
                except Exception as e:
                    print(f"Warning: Failed to render pushover animation: {e}")
                    
            plt.show()
        
        ops.wipe()
        return Iffinish, currentDisp
        
    def _post_process_pushover(self, disp_file, base_disp_file, base_reaction_file):
        """解析静力推覆记录，提取基底剪力及 V/W 系数，并保存位移与层间位移角时程。"""
        try:
            disp_data = np.loadtxt(disp_file)
        except Exception:
            return

        if len(disp_data) == 0:
            return

        if disp_data.ndim == 1:
            disp_data = disp_data.reshape(1, -1)

        Times = disp_data[:, 0]
        Disps = disp_data[:, 1:]
        
        self.BaseShearHistory = np.zeros(len(Times))
        if base_reaction_file and base_reaction_file.exists():
            try:
                react_data = np.loadtxt(base_reaction_file)
                if react_data.ndim == 1: react_data = react_data.reshape(1, -1)
                self.BaseShearHistory = -np.sum(react_data[:, 1:], axis=1)
            except:
                pass
        if self.TotalWeight > 0.0:
            self.BaseShearCoefficientHistory = self.BaseShearHistory / self.TotalWeight
        else:
            self.BaseShearCoefficientHistory = np.zeros(len(Times))
        
        BaseDisps = np.zeros(len(Times))
        if base_disp_file and base_disp_file.exists():
            try:
                base_data = np.loadtxt(base_disp_file)
                if base_data.ndim == 1: base_data = base_data.reshape(1, -1)
                BaseDisps = np.mean(base_data[:, 1:], axis=1) if base_data.shape[1] > 1 else np.zeros(len(Times))
            except:
                pass
                
        num_floors = Disps.shape[1]
        
        self.DriftHistory = {}
        self.NodeDispHistory = {}
        
        self.NodeDispHistory['time'] = Times
        self.DriftHistory['time'] = Times
        
        for i in range(num_floors):
            lower_disp = BaseDisps if i == 0 else Disps[:, i-1]
            drift_i = (Disps[:, i] - lower_disp) / self._story_heights[i]
            
            self.NodeDispHistory[i + 1] = Disps[:, i]
            self.DriftHistory[i + 1] = drift_i

    def ModalAnalysis(self, num_modes: int = 5, ifprint: bool = True) -> tuple[List[float], List[float]]:
        """
        对当前模型进行模态分析。
        
        Parameters
        ----------
        num_modes : int, optional
            需要计算的模态阶数，默认前 5 阶。
        ifprint : bool, optional
            是否在控制台打印前两阶的周期。
            
        Returns
        -------
        Tuple[List[float], List[float]]
            返回元组 (omegas, periods)，分别为圆频率列表和周期列表。
        """
        import math
        
        # 为了能够在各种荷载建立前后稳定调用特征值分析，先清除可能遗留的分析配置
        ops.wipeAnalysis()

        # define a minimal analysis setup to avoid OpenSeesPy warnings and to accommodate 'eigen' command
        ops.system('BandGeneral')
        ops.numberer('RCM')
        ops.constraints('Transformation')
        ops.test('NormDispIncr', 1.0e-12, 10, 0)
        ops.algorithm('Newton')
        ops.integrator('LoadControl', 0.0)
        ops.analysis('Static')

        eigen_values = ops.eigen(num_modes)
        
        # 清除静力分析配置避免干扰后续的瞬态分析
        ops.wipeAnalysis()
        if isinstance(eigen_values, (int, float)):
            eigen_values = [eigen_values]

        omegas = [math.sqrt(value) for value in eigen_values]
        periods = [2.0 * math.pi / omega for omega in omegas]
        
        if ifprint and len(periods) >= 2:
            print(f'Eigen Analysis: T1 = {periods[0]:.2f} s; T2 = {periods[1]:.2f} s')
        elif ifprint and len(periods) == 1:
            print(f'Eigen Analysis: T1 = {periods[0]:.2f} s')

        return omegas, periods


    def PlotModel(self, **kwargs):
        """
        使用 opsvis 包绘制包含节点和单元的模型示意图。
        如果当前模型尚未建立，会自动调用 build_model_func 进行建立再绘制。
        
        Parameters
        ----------
        kwargs : dict
            其他直接传递给 opsvis.plot_model() 的参数 (例如: node_labels=1, element_labels=1)。
        """
            
        ops.wipe()
        self.build_model_func()
        
        # 绘制 OpenSees 模型
        opsv.plot_model(gauss_points=False,**kwargs)
        plt.show()

    def Plot_Gravity_def(self, **kwargs):
        """
        使用 opsvis 包绘制施加自动重力后的结构变形图。
        会自动调用 build_model_func 建立模型，然后运行施加重力，最后显示变形。
        
        Parameters
        ----------
        kwargs : dict
            其他直接传递给 opsvis.plot_defo() 的参数 (例如: fmt_defo='b-', fmt_undefo='r--')。
        """
        ops.wipe()
        
        # 1. 建立模型并施加重力
        self.build_model_func()
        self._auto_apply_gravity()
        
        # 2. 绘制重力作用下的变形图
        opsv.plot_defo(**kwargs)
        plt.show()

    def Plot_Mode_Shape(self, mode_no: int = 1, **kwargs):
        """
        使用 opsvis 包绘制指定阶数的设计模态振型图。
        会自动调用 build_model_func 建立模型，运行特征值分析，最后显示振型。
        
        Parameters
        ----------
        mode_no : int, optional
            要绘制的振型阶数，默认为 1。
        kwargs : dict
            其他直接传递的画图参数。
        """
        ops.wipe()
        self.build_model_func()
        self.ModalAnalysis(num_modes=max(mode_no, 2))
        
        # 绘制模态振型
        # 自动计算结构边界区域，防止因为动图未给边界而退化为极小范围
        if 'xlim' not in kwargs or 'ylim' not in kwargs:
            node_tags = ops.getNodeTags()
            if node_tags:
                coords = np.array([ops.nodeCoord(tag) for tag in node_tags])
                if coords.size > 0:
                    xmin, xmax = np.min(coords[:, 0]), np.max(coords[:, 0])
                    ymin, ymax = np.min(coords[:, 1]), np.max(coords[:, 1])
                    
                    dx = max(xmax - xmin, 1.0)
                    dy = max(ymax - ymin, 1.0)
                    
                    if 'xlim' not in kwargs:
                        # 振型变形可能会横向放大，为了不越界，留余量（比如给高度的50%作为左右余地）
                        kwargs['xlim'] = [xmin - dy * 0.5, xmax + dy * 0.5]
                    if 'ylim' not in kwargs:
                        # 上下留适度余量
                        kwargs['ylim'] = [ymin - dy * 0.1, ymax + dy * 0.1]

        # 必须返回参数防止垃圾回收
        anim = opsv.anim_mode(modeNo=mode_no, **kwargs)

        plt.show()
