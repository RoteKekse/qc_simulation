PATH=geometry/N2/
FILE=n2
BASIS=cc-pvtz
SHIFT=135
K1=200
K=50
H=1.0

build/hf.out $PATH $FILE $BASIS
build/get_1elec_int.out $PATH $FILE $BASIS
build/get_2elec_int.out $PATH $FILE $BASIS
build/basistransform.out $FILE $BASIS

build/buildhamil.out $FILE $BASIS $SHIFT
build/buildhamildiag.out $FILE $BASIS $SHIFT
build/buildprecon.out $FILE $BASIS $SHIFT $K1 $K2 $H
build/buildprecon2.out $FILE $BASIS $SHIFT 
