import openseespy.opensees as ops

X_COORDS = [-9000.0, -7500.0, 0.0, 7500.0, 9000.0]
Y_COORDS = [0.0, 5000.0, 9000.0, 13000.0, 17000.0, 21000.0, 25000.0]
P_LOADS = [28126.0, 168756.0, 281260.0, 168756.0, 28126.0]

G = 9800.0
FY = 460.0
ES = 200000.0
STEEL_B = 0.01
NP_INTEGRATION = 5

MAT_ID_STEEL = 11
COL_SEC_TAGS = [None, 11, 12, 13, 14, 15, 16]
BEAM_SEC_TAGS = [None, 21, 22, 23, 24, 25, 26]

COL_TRANSF_TAG = 1
BEAM_TRANSF_TAG = 2

FLOOR_NODES = [103, 203, 303, 403, 503, 603]
STORY_HEIGHTS = [5000.0, 4000.0, 4000.0, 4000.0, 4000.0, 4000.0]
BASE_NODES = [1, 2, 3, 4, 5]


def create_frame_nodes_and_constraints() -> None:
    for story_idx, y in enumerate(Y_COORDS):
        for col_idx, x in enumerate(X_COORDS, start=1):
            node_tag = story_idx * 100 + col_idx
            ops.node(node_tag, x, y)

    for node_tag in range(1, 6):
        ops.fix(node_tag, 1, 1, 1)

    for story in range(1, 7):
        master_node = story * 100 + 3
        for col in range(1, 6):
            if col == 3:
                continue
            slave_node = story * 100 + col
            ops.equalDOF(master_node, slave_node, 1)


def define_w_section(sec_id: int, mat_id: int, d: float, bf: float, tf: float, tw: float,
                     nfdw: int, nftw: int, nfbf: int, nftf: int) -> None:
    dw = d - 2.0 * tf
    y1 = -d / 2.0
    y2 = -dw / 2.0
    y3 = dw / 2.0
    y4 = d / 2.0
    z1 = -bf / 2.0
    z2 = -tw / 2.0
    z3 = tw / 2.0
    z4 = bf / 2.0

    ops.section("Fiber", sec_id)
    ops.patch("rect", mat_id, nftf, nfbf, y1, z1, y2, z4)
    ops.patch("rect", mat_id, nfdw, nftw, y2, z2, y3, z3)
    ops.patch("rect", mat_id, nftf, nfbf, y3, z1, y4, z4)


def define_materials_and_sections() -> None:
    ops.uniaxialMaterial("Steel01", MAT_ID_STEEL, FY, ES, STEEL_B)

    nfdw, nftw, nfbf, nftf = 10, 5, 15, 5

    define_w_section(11, MAT_ID_STEEL, 650.0, 550.0, 24.0, 14.0, nfdw, nftw, nfbf, nftf)
    define_w_section(12, MAT_ID_STEEL, 650.0, 450.0, 24.0, 14.0, nfdw, nftw, nfbf, nftf)
    define_w_section(13, MAT_ID_STEEL, 550.0, 350.0, 24.0, 14.0, nfdw, nftw, nfbf, nftf)
    define_w_section(14, MAT_ID_STEEL, 450.0, 350.0, 24.0, 14.0, nfdw, nftw, nfbf, nftf)
    define_w_section(15, MAT_ID_STEEL, 450.0, 250.0, 24.0, 14.0, nfdw, nftw, nfbf, nftf)
    define_w_section(16, MAT_ID_STEEL, 450.0, 250.0, 24.0, 14.0, nfdw, nftw, nfbf, nftf)

    define_w_section(21, MAT_ID_STEEL, 450.0, 200.0, 24.0, 16.0, nfdw, nftw, nfbf, nftf)
    define_w_section(22, MAT_ID_STEEL, 450.0, 200.0, 24.0, 16.0, nfdw, nftw, nfbf, nftf)
    define_w_section(23, MAT_ID_STEEL, 450.0, 200.0, 24.0, 16.0, nfdw, nftw, nfbf, nftf)
    define_w_section(24, MAT_ID_STEEL, 450.0, 200.0, 20.0, 16.0, nfdw, nftw, nfbf, nftf)
    define_w_section(25, MAT_ID_STEEL, 400.0, 200.0, 20.0, 16.0, nfdw, nftw, nfbf, nftf)
    define_w_section(26, MAT_ID_STEEL, 400.0, 200.0, 20.0, 16.0, nfdw, nftw, nfbf, nftf)


def define_geometric_transformations() -> None:
    ops.geomTransf("PDelta", COL_TRANSF_TAG)
    ops.geomTransf("Linear", BEAM_TRANSF_TAG)


def create_columns() -> None:
    for story in range(1, 7):
        sec_tag = COL_SEC_TAGS[story]
        for col in range(1, 6):
            ele_tag = 10000 + 100 * story + col
            node_i = (story - 1) * 100 + col
            node_j = story * 100 + col
            ops.beamIntegration("Lobatto", ele_tag, sec_tag, NP_INTEGRATION)
            ops.element("forceBeamColumn", ele_tag, node_i, node_j, COL_TRANSF_TAG, ele_tag)


def create_standard_beams() -> None:
    for story in range(1, 7):
        sec_tag = BEAM_SEC_TAGS[story]
        for bay in range(1, 5):
            ele_tag = 20000 + 100 * story + bay
            node_i = story * 100 + bay
            node_j = story * 100 + bay + 1
            ops.beamIntegration("Lobatto", ele_tag, sec_tag, NP_INTEGRATION)
            ops.element("forceBeamColumn", ele_tag, node_i, node_j, BEAM_TRANSF_TAG, ele_tag)


def assign_story_masses(master_y_mass: float, slave_y_mass: float) -> None:
    for story in range(1, 7):
        total_mass = sum(load / G for load in P_LOADS)
        master_node = story * 100 + 3
        ops.mass(master_node, total_mass, master_y_mass, 1.0e-9)
        for col in range(1, 6):
            if col == 3:
                continue
            ops.mass(story * 100 + col, 0.0, slave_y_mass, 1.0e-9)


def build_model() -> None:
    ops.model("basic", "-ndm", 2, "-ndf", 3)
    create_frame_nodes_and_constraints()
    assign_story_masses(master_y_mass=1.0e-9, slave_y_mass=1.0e-9)
    define_materials_and_sections()
    define_geometric_transformations()
    create_columns()
    create_standard_beams()