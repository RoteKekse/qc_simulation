GEOM=geometry/H2O/
BASIS=double
SHIFT=25
rank=32
NOB=24
eps=1e-6
DATA=FCIDUMP.h2o_24

DMRG_ITER=2

PPGD_ITER=2
PPGD_OPT=1
PPGD_STEP=3.0
PPGD_PRE_RANK=1

#mkdir -p results/$PATH


#rm data/*
FILE=h2o_48

echo $FILE

build/read_FCIdump.out $FILE $BASIS $DATA $NOB

build/buildhamil.out $FILE $BASIS 0


build/buildprecon.out $FILE $BASIS $SHIFT 1 1
build/buildprecon.out $FILE $BASIS $SHIFT 2 1
build/buildprecon.out $FILE $BASIS $SHIFT 3 1


build/dmrg.out $FILE $BASIS 0 5 5 $eps 1

build/ppgd.out $FILE $BASIS 0 0 $rank $PPGD_ITER $eps $PPGD_OPT $PPGD_STEP

build/ppgd.out $FILE $BASIS 1 0 $rank $PPGD_ITER $eps $PPGD_OPT $PPGD_STEP

build/ppgd.out $FILE $BASIS 2 0 $rank $PPGD_ITER $eps $PPGD_OPT $PPGD_STEP

build/ppgd.out $FILE $BASIS 3 0 $rank $PPGD_ITER $eps $PPGD_OPT $PPGD_STEP

build/dmrg.out $FILE $BASIS 0 $rank $DMRG_ITER $eps 0


