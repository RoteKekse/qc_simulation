GEOM=geometry/H2O/
BASIS=double
SHIFT=77
rank=32 
NOB=24
eps=1e-6
DATA=FCIDUMP.h2o_24

DMRG_ITER=15

PPGD_ITER=30
PPGD_OPT=0
PPGD_STEP=3.0
PPGD_PRE_RANK=1

#mkdir -p results/$PATH


#rm data/*
FILE=h2o_48

echo $FILE

#build/read_FCIdump.out $FILE $BASIS $DATA $NOB

#build/buildhamil.out $FILE $BASIS 0


#build/dmrg.out $FILE $BASIS 0 3 5 $eps 1


#build/buildprecon.out $FILE $BASIS $SHIFT 1 2

#build/ppgd.out $FILE $BASIS 1 0 32 $PPGD_ITER $eps 0 $PPGD_STEP
#build/dmrg.out $FILE $BASIS 0 32 $DMRG_ITER $eps 0

#build/ppgd.out $FILE $BASIS 1 0 64 $PPGD_ITER $eps 0 $PPGD_STEP
#build/dmrg.out $FILE $BASIS 0 64 $DMRG_ITER $eps 0

#build/ppgd.out $FILE $BASIS 1 0 128 $PPGD_ITER $eps 0 $PPGD_STEP
build/dmrg.out $FILE $BASIS 0 128 $DMRG_ITER $eps 0
