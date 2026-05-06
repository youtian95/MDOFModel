import os

def ReadRecord(inFilename, outFilename):
    # 查找 inFilename。若文件扩展名为 .at2，调用 ReadRecord_PEER；
    # 若为 .txt（第1列为时间，第2列为加速度），调用 ReadRecord_TXT。
    #
    # 参数:
    #   inFilename: 不含扩展名的文件路径
    #   outFilename: 输出文件路径，扩展名为 '.dat'
    if os.path.exists(inFilename + '.at2'):
        dt, npts = ReadRecord_PEER (inFilename + '.at2', outFilename)
    elif os.path.exists(inFilename + '.txt'):
        dt, npts = ReadRecord_TXT (inFilename + '.txt', outFilename)
    else:
        print('ERROR: Cant find record file!')
        dt = None
        npts = None
    return dt, npts

def ReadRecord_TXT (inFilename, outFilename):
    
    inFileID = open(inFilename, 'r')
    outFileID = open(outFilename, 'w')

    time_1 = 0.0
    time_2 = 0.0
    npts=0
    for line in inFileID:
        if line == '\n':
            continue
        else:
            words = str.replace(line,',',' ').split()
            lengthLine = len(words)
            if lengthLine == 2:
                npts+=1
                if npts==1:
                    time_1=float(words[0])
                elif npts==2:
                    time_2=float(words[0])
                outFileID.write(words[1])
                outFileID.write('\n')
    dt = time_2 - time_1
    
    inFileID.close()
    outFileID.close()

    return dt, npts

def ReadRecord_PEER (inFilename, outFilename):
    # 从 PEER 强震数据库的地震动记录文件中提取时间步长，并将加速度数据写入输出文件。
    #
    # 形式参数:
    #   inFilename  -- 包含 PEER 强震记录的输入文件
    #   outFilename -- 输出文件，供 OpenSees 读取
    # 返回值:
    #   dt   -- 由文件头确定的时间步长
    #   nPts -- 由文件头确定的数据点数
    #
    # 支持的文件头格式（以下两种之一）：
    #  1) 新版 NGA 格式
    #	 PACIFIC ENGINEERING AND ANALYSIS STRONG-MOTION DATA
    #	  IMPERIAL VALLEY 10/15/79 2319, EL CENTRO ARRAY 6, 230
    #	  ACCELERATION TIME HISTORY IN UNITS OF G
    #	  3930 0.00500 NPTS, DT
    #
    #  2) 旧版 SMD 格式
    #	 PACIFIC ENGINEERING AND ANALYSIS STRONG-MOTION DATA
    #	  IMPERIAL VALLEY 10/15/79 2319, EL CENTRO ARRAY 6, 230
    #	  ACCELERATION TIME HISTORY IN UNITS OF G
    #	  NPTS=  3930, DT= .00500 SEC

    dt = 0.0
    npts = 0
    
    # 打开输入文件
    inFileID = open(inFilename, 'r')
    
    # 打开输出文件
    outFileID = open(outFilename, 'w')
	
    # 标志位：已找到 dt 后开始读取加速度数据（假设 dt 在文件头最后一行）
    flag = 0
	
    # 逐行读取
    for line in inFileID:
        if line == '\n':
            # 空行，跳过
            continue
        elif flag == 1:
            # 将加速度数据写入输出文件
            outFileID.write(line)
        else:
            # 在文件头中查找 dt
            words = line.split()
            lengthLine = len(words)

            if lengthLine >= 4:

                if words[0] == 'NPTS=':
                    # 旧版 SMD 格式
                    for word in words:
                        if word != '':
                            # 读取时间步长
                            if flag == 1:
                                dt = float(word)
                                break

                            if flag == 2:
                                npts = int(word.strip(','))
                                flag = 0

                            # 找到目标关键词并设置标志位
                            if word == 'DT=' or word == 'dt':
                                flag = 1

                            if word == 'NPTS=':
                                flag = 2
                        
                    
                elif words[-1] == 'DT':
                    # 新版 NGA 格式
                    count = 0
                    for word in words:
                        if word != '':
                            if count == 0:
                                npts = int(word)
                            elif count == 1:
                                dt = float(word)
                            elif word == 'DT':
                                flag = 1
                                break

                            count += 1

                        

    inFileID.close()
    outFileID.close()

    return dt, npts
