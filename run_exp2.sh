GEOM=geometry/N2/
BASIS=cc-pvtz
SHIFT=110
rank=32
eps=1e-6

DMRG_ITER=40

PPGD_ITER=70
PPGD_OPT=1
PPGD_STEP=4.0
PPGD_PRE_RANK=1

#mkdir -p results/$PATH


#rm data/*
FILE=n2_1.1

echo $FILE
build/hf.out $GEOM $FILE $BASIS
build/get_1elec_int.out $GEOM $FILE $BASIS
build/get_2elec_int.out $GEOM $FILE $BASIS
build/basistransform.out $FILE $BASIS

build/buildhamil.out $FILE $BASIS 0


build/buildprecon.out $FILE $BASIS $SHIFT 1 1
build/buildprecon.out $FILE $BASIS $SHIFT 2 1
build/buildprecon.out $FILE $BASIS $SHIFT 3 1


build/dmrg.out $FILE $BASIS 0 3 5 $eps 1

#build/ppgd.out $FILE $BASIS 0 0 $rank $PPGD_ITER $eps $PPGD_OPT $PPGD_STEP
#build/ppgd.out $FILE $BASIS 1 0 16 $PPGD_ITER $eps $PPGD_OPT $PPGD_STEP
build/ppgd.out $FILE $BASIS 2 0 32 $PPGD_ITER $eps $PPGD_OPT $PPGD_STEP
#build/ppgd.out $FILE $BASIS 1 0 48 $PPGD_ITER $eps $PPGD_OPT $PPGD_STEP

#build/ppgd.out $FILE $BASIS 2 0 $rank $PPGD_ITER $eps $PPGD_OPT $PPGD_STEP

#build/ppgd.out $FILE $BASIS 3 0 $rank $PPGD_ITER $eps $PPGD_OPT $PPGD_STEP

#build/dmrg.out $FILE $BASIS 0 16 $DMRG_ITER $eps 0
build/dmrg.out $FILE $BASIS 0 32 $DMRG_ITER $eps 0
#build/dmrg.out $FILE $BASIS 0 48 $DMRG_ITER $eps 0


