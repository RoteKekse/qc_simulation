PATH=geometry/N2/
FILE=n2
BASIS=cc-pvtz
SHIFT=110
K=700

build/hf.out $PATH $FILE $BASIS
build/get_1elec_int.out $PATH $FILE $BASIS
build/get_2elec_int.out $PATH $FILE $BASIS
build/basistransform.out $FILE $BASIS

build/buildHamil.out $FILE $BASIS
build/buildHamilDiag.out $FILE $BASIS
build/buildprecon.out $FILE $BASIS $SHIFT $K
