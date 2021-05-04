PATH=geometry/N2/
FILE=n2
BASIS=cc-pvtz
SHIFT=110
K=700
H=0.05

build/hf.out $PATH $FILE $BASIS
build/get_1elec_int.out $PATH $FILE $BASIS
build/get_2elec_int.out $PATH $FILE $BASIS
build/basistransform.out $FILE $BASIS

build/buildhamil.out $FILE $BASIS
build/buildhamildiag.out $FILE $BASIS
build/buildprecon.out $FILE $BASIS $SHIFT $K $H
