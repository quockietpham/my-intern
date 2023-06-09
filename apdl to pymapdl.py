import matplotlib.pyplot as plt

from ansys.mapdl.core import launch_mapdl
mapdl = launch_mapdl()
mapdl.run("finish ")
mapdl.run("/clear ")
mapdl.run("/PREP7   !vào Preprocessor")
mapdl.run("! Khai báo biến")
mapdl.run("EE=2e11")
mapdl.run("nu=0.3")
mapdl.run("A1=1E-4")
mapdl.run("A2=2E-4")
mapdl.run("FF=10000")
mapdl.run("LL=1")
mapdl.run("*AFUN,DEG")

mapdl.run("ET,1,LINK180	!khai báo phần tử LINK180")
mapdl.run("MP,EX,1,EE	!khai báo module đàn hồi E")
mapdl.run("MP,PRXY,1,nu	!hệ số poisson")
mapdl.run("R,1,A1	!khai báo diện tích mặt cắt 1 cm2")
mapdl.run("R,2,A2")

mapdl.run("! Mo hinh")
mapdl.run("N,1,0,0,0")
mapdl.run("N,2,LL*TAN(30)")
mapdl.run("N,3,-LL*TAN(30)")
mapdl.run("N,4,0,-LL")

mapdl.run("REAL,1")
mapdl.run("E,4,3")
mapdl.run("E,4,2")
mapdl.run("REAL,2")
mapdl.run("E,4,1")

mapdl.run("! Điều kiện biên & tải trọng")
mapdl.run("D,1,ALL,0")
mapdl.run("D,2,ALL,0")
mapdl.run("D,3,ALL,0")
mapdl.run("F,4,FY,-FF")

mapdl.run("/SOLU")
mapdl.run("SOLVE")

mapdl.run("/POST1	!vào General Postproc")
mapdl.run("PLNSOL, U,SUM, 0,1.0	!Xuất thành phần chuyển vị tổng")
mapdl.run("ETABLE,NOILUC,SMISC, 1	!Tạo bảng tính nội lực")
# mapdl.run("PLLS,NOILUC,NOILUC,1,0	!Xuất kết quả nội lực")
# mapdl.run("ETABLE,UNGSUAT,LS, 1	!Tạo bảng tính ứng suất")
# mapdl.run("PLETAB,UNGSUAT,NOAV		!Xuất kết quả ứng suất")
mapdl.exit()
