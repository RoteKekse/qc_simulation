PATH=geometry/N2/
FILE=n2
BASIS=cc-pvtz

build/hf.out $PATH $FILE $BASIS
build/get_1elec_int.out $PATH $FILE $BASIS
build/get_2elec_int.out $PATH $FILE $BASIS
build/basistransform.out $FILE $BASIS
