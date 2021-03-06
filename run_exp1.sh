GEOM=geometry/N2/
BASIS=cc-pvtz
SHIFT=110
rank=32
eps=1e-6

DMRG_ITER=15

PPGD_ITER=25
PPGD_OPT=0
PPGD_STEP=4.0
PPGD_PRE_RANK=1

#mkdir -p results/$PATH

for FILE in ${GEOM}*; do 
    if [ -f "$FILE" ]; then 
    	rm data/*
    	FILE=${FILE%.xyz}
    	FILE=${FILE#$GEOM};
    	echo $FILE
	build/hf.out $GEOM $FILE $BASIS
	build/get_1elec_int.out $GEOM $FILE $BASIS
	build/get_2elec_int.out $GEOM $FILE $BASIS
	build/basistransform.out $FILE $BASIS

	build/buildhamil.out $FILE $BASIS 0
	#build/buildhamildiag.out $FILE $BASIS $SHIFT

	build/buildprecon.out $FILE $BASIS $SHIFT 1 1


	build/dmrg.out $FILE $BASIS 0 3 5 $eps 1
	build/dmrg.out $FILE $BASIS 0 $rank $DMRG_ITER $eps 0
	build/ppgd.out $FILE $BASIS $PPGD_PRE_RANK 0 $rank $PPGD_ITER $eps $PPGD_OPT $PPGD_STEP
        
    fi 
done




