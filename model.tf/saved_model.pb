??
??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
?
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

?
Conv2DBackpropInput
input_sizes
filter"T
out_backprop"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.5.02v2.5.0-rc3-213-ga4dfb8d1a718Ѹ
?
encoder_layer_extra1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_nameencoder_layer_extra1/kernel
?
/encoder_layer_extra1/kernel/Read/ReadVariableOpReadVariableOpencoder_layer_extra1/kernel*&
_output_shapes
:*
dtype0
?
encoder_layer_extra1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_nameencoder_layer_extra1/bias
?
-encoder_layer_extra1/bias/Read/ReadVariableOpReadVariableOpencoder_layer_extra1/bias*
_output_shapes
:*
dtype0
?
encoder_layer_extra2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_nameencoder_layer_extra2/kernel
?
/encoder_layer_extra2/kernel/Read/ReadVariableOpReadVariableOpencoder_layer_extra2/kernel*&
_output_shapes
:*
dtype0
?
encoder_layer_extra2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_nameencoder_layer_extra2/bias
?
-encoder_layer_extra2/bias/Read/ReadVariableOpReadVariableOpencoder_layer_extra2/bias*
_output_shapes
:*
dtype0
?
encoder_layer_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameencoder_layer_1/kernel
?
*encoder_layer_1/kernel/Read/ReadVariableOpReadVariableOpencoder_layer_1/kernel*&
_output_shapes
:*
dtype0
?
encoder_layer_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameencoder_layer_1/bias
y
(encoder_layer_1/bias/Read/ReadVariableOpReadVariableOpencoder_layer_1/bias*
_output_shapes
:*
dtype0
s
code/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*
shared_namecode/kernel
l
code/kernel/Read/ReadVariableOpReadVariableOpcode/kernel*
_output_shapes
:	?*
dtype0
j
	code/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name	code/bias
c
code/bias/Read/ReadVariableOpReadVariableOp	code/bias*
_output_shapes
:*
dtype0
?
decoder_layer_extra1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namedecoder_layer_extra1/kernel
?
/decoder_layer_extra1/kernel/Read/ReadVariableOpReadVariableOpdecoder_layer_extra1/kernel*&
_output_shapes
:*
dtype0
?
decoder_layer_extra1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_namedecoder_layer_extra1/bias
?
-decoder_layer_extra1/bias/Read/ReadVariableOpReadVariableOpdecoder_layer_extra1/bias*
_output_shapes
:*
dtype0
?
decoder_layer_extra2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namedecoder_layer_extra2/kernel
?
/decoder_layer_extra2/kernel/Read/ReadVariableOpReadVariableOpdecoder_layer_extra2/kernel*&
_output_shapes
:*
dtype0
?
decoder_layer_extra2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_namedecoder_layer_extra2/bias
?
-decoder_layer_extra2/bias/Read/ReadVariableOpReadVariableOpdecoder_layer_extra2/bias*
_output_shapes
:*
dtype0
?
decoder_layer_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_namedecoder_layer_1/kernel
?
*decoder_layer_1/kernel/Read/ReadVariableOpReadVariableOpdecoder_layer_1/kernel*&
_output_shapes
:*
dtype0
?
decoder_layer_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_namedecoder_layer_1/bias
y
(decoder_layer_1/bias/Read/ReadVariableOpReadVariableOpdecoder_layer_1/bias*
_output_shapes
:*
dtype0
?
decoder_layer_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_namedecoder_layer_2/kernel
?
*decoder_layer_2/kernel/Read/ReadVariableOpReadVariableOpdecoder_layer_2/kernel*&
_output_shapes
:*
dtype0
?
decoder_layer_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_namedecoder_layer_2/bias
y
(decoder_layer_2/bias/Read/ReadVariableOpReadVariableOpdecoder_layer_2/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
?
"Adam/encoder_layer_extra1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/encoder_layer_extra1/kernel/m
?
6Adam/encoder_layer_extra1/kernel/m/Read/ReadVariableOpReadVariableOp"Adam/encoder_layer_extra1/kernel/m*&
_output_shapes
:*
dtype0
?
 Adam/encoder_layer_extra1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Adam/encoder_layer_extra1/bias/m
?
4Adam/encoder_layer_extra1/bias/m/Read/ReadVariableOpReadVariableOp Adam/encoder_layer_extra1/bias/m*
_output_shapes
:*
dtype0
?
"Adam/encoder_layer_extra2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/encoder_layer_extra2/kernel/m
?
6Adam/encoder_layer_extra2/kernel/m/Read/ReadVariableOpReadVariableOp"Adam/encoder_layer_extra2/kernel/m*&
_output_shapes
:*
dtype0
?
 Adam/encoder_layer_extra2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Adam/encoder_layer_extra2/bias/m
?
4Adam/encoder_layer_extra2/bias/m/Read/ReadVariableOpReadVariableOp Adam/encoder_layer_extra2/bias/m*
_output_shapes
:*
dtype0
?
Adam/encoder_layer_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_nameAdam/encoder_layer_1/kernel/m
?
1Adam/encoder_layer_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/encoder_layer_1/kernel/m*&
_output_shapes
:*
dtype0
?
Adam/encoder_layer_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_nameAdam/encoder_layer_1/bias/m
?
/Adam/encoder_layer_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/encoder_layer_1/bias/m*
_output_shapes
:*
dtype0
?
Adam/code/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*#
shared_nameAdam/code/kernel/m
z
&Adam/code/kernel/m/Read/ReadVariableOpReadVariableOpAdam/code/kernel/m*
_output_shapes
:	?*
dtype0
x
Adam/code/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameAdam/code/bias/m
q
$Adam/code/bias/m/Read/ReadVariableOpReadVariableOpAdam/code/bias/m*
_output_shapes
:*
dtype0
?
"Adam/decoder_layer_extra1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/decoder_layer_extra1/kernel/m
?
6Adam/decoder_layer_extra1/kernel/m/Read/ReadVariableOpReadVariableOp"Adam/decoder_layer_extra1/kernel/m*&
_output_shapes
:*
dtype0
?
 Adam/decoder_layer_extra1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Adam/decoder_layer_extra1/bias/m
?
4Adam/decoder_layer_extra1/bias/m/Read/ReadVariableOpReadVariableOp Adam/decoder_layer_extra1/bias/m*
_output_shapes
:*
dtype0
?
"Adam/decoder_layer_extra2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/decoder_layer_extra2/kernel/m
?
6Adam/decoder_layer_extra2/kernel/m/Read/ReadVariableOpReadVariableOp"Adam/decoder_layer_extra2/kernel/m*&
_output_shapes
:*
dtype0
?
 Adam/decoder_layer_extra2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Adam/decoder_layer_extra2/bias/m
?
4Adam/decoder_layer_extra2/bias/m/Read/ReadVariableOpReadVariableOp Adam/decoder_layer_extra2/bias/m*
_output_shapes
:*
dtype0
?
Adam/decoder_layer_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_nameAdam/decoder_layer_1/kernel/m
?
1Adam/decoder_layer_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/decoder_layer_1/kernel/m*&
_output_shapes
:*
dtype0
?
Adam/decoder_layer_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_nameAdam/decoder_layer_1/bias/m
?
/Adam/decoder_layer_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/decoder_layer_1/bias/m*
_output_shapes
:*
dtype0
?
Adam/decoder_layer_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_nameAdam/decoder_layer_2/kernel/m
?
1Adam/decoder_layer_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/decoder_layer_2/kernel/m*&
_output_shapes
:*
dtype0
?
Adam/decoder_layer_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_nameAdam/decoder_layer_2/bias/m
?
/Adam/decoder_layer_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/decoder_layer_2/bias/m*
_output_shapes
:*
dtype0
?
"Adam/encoder_layer_extra1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/encoder_layer_extra1/kernel/v
?
6Adam/encoder_layer_extra1/kernel/v/Read/ReadVariableOpReadVariableOp"Adam/encoder_layer_extra1/kernel/v*&
_output_shapes
:*
dtype0
?
 Adam/encoder_layer_extra1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Adam/encoder_layer_extra1/bias/v
?
4Adam/encoder_layer_extra1/bias/v/Read/ReadVariableOpReadVariableOp Adam/encoder_layer_extra1/bias/v*
_output_shapes
:*
dtype0
?
"Adam/encoder_layer_extra2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/encoder_layer_extra2/kernel/v
?
6Adam/encoder_layer_extra2/kernel/v/Read/ReadVariableOpReadVariableOp"Adam/encoder_layer_extra2/kernel/v*&
_output_shapes
:*
dtype0
?
 Adam/encoder_layer_extra2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Adam/encoder_layer_extra2/bias/v
?
4Adam/encoder_layer_extra2/bias/v/Read/ReadVariableOpReadVariableOp Adam/encoder_layer_extra2/bias/v*
_output_shapes
:*
dtype0
?
Adam/encoder_layer_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_nameAdam/encoder_layer_1/kernel/v
?
1Adam/encoder_layer_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/encoder_layer_1/kernel/v*&
_output_shapes
:*
dtype0
?
Adam/encoder_layer_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_nameAdam/encoder_layer_1/bias/v
?
/Adam/encoder_layer_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/encoder_layer_1/bias/v*
_output_shapes
:*
dtype0
?
Adam/code/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*#
shared_nameAdam/code/kernel/v
z
&Adam/code/kernel/v/Read/ReadVariableOpReadVariableOpAdam/code/kernel/v*
_output_shapes
:	?*
dtype0
x
Adam/code/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameAdam/code/bias/v
q
$Adam/code/bias/v/Read/ReadVariableOpReadVariableOpAdam/code/bias/v*
_output_shapes
:*
dtype0
?
"Adam/decoder_layer_extra1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/decoder_layer_extra1/kernel/v
?
6Adam/decoder_layer_extra1/kernel/v/Read/ReadVariableOpReadVariableOp"Adam/decoder_layer_extra1/kernel/v*&
_output_shapes
:*
dtype0
?
 Adam/decoder_layer_extra1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Adam/decoder_layer_extra1/bias/v
?
4Adam/decoder_layer_extra1/bias/v/Read/ReadVariableOpReadVariableOp Adam/decoder_layer_extra1/bias/v*
_output_shapes
:*
dtype0
?
"Adam/decoder_layer_extra2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/decoder_layer_extra2/kernel/v
?
6Adam/decoder_layer_extra2/kernel/v/Read/ReadVariableOpReadVariableOp"Adam/decoder_layer_extra2/kernel/v*&
_output_shapes
:*
dtype0
?
 Adam/decoder_layer_extra2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Adam/decoder_layer_extra2/bias/v
?
4Adam/decoder_layer_extra2/bias/v/Read/ReadVariableOpReadVariableOp Adam/decoder_layer_extra2/bias/v*
_output_shapes
:*
dtype0
?
Adam/decoder_layer_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_nameAdam/decoder_layer_1/kernel/v
?
1Adam/decoder_layer_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/decoder_layer_1/kernel/v*&
_output_shapes
:*
dtype0
?
Adam/decoder_layer_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_nameAdam/decoder_layer_1/bias/v
?
/Adam/decoder_layer_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/decoder_layer_1/bias/v*
_output_shapes
:*
dtype0
?
Adam/decoder_layer_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_nameAdam/decoder_layer_2/kernel/v
?
1Adam/decoder_layer_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/decoder_layer_2/kernel/v*&
_output_shapes
:*
dtype0
?
Adam/decoder_layer_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_nameAdam/decoder_layer_2/bias/v
?
/Adam/decoder_layer_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/decoder_layer_2/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
?\
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?[
value?[B?[ B?[
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer-4
layer_with_weights-3
layer-5
layer-6
layer_with_weights-4
layer-7
	layer_with_weights-5
	layer-8

layer_with_weights-6

layer-9
layer_with_weights-7
layer-10
	optimizer
trainable_variables
	variables
regularization_losses
	keras_api

signatures
R
trainable_variables
	variables
regularization_losses
	keras_api
h

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
h

kernel
bias
trainable_variables
	variables
 regularization_losses
!	keras_api
h

"kernel
#bias
$trainable_variables
%	variables
&regularization_losses
'	keras_api
R
(trainable_variables
)	variables
*regularization_losses
+	keras_api
h

,kernel
-bias
.trainable_variables
/	variables
0regularization_losses
1	keras_api
R
2trainable_variables
3	variables
4regularization_losses
5	keras_api
h

6kernel
7bias
8trainable_variables
9	variables
:regularization_losses
;	keras_api
h

<kernel
=bias
>trainable_variables
?	variables
@regularization_losses
A	keras_api
h

Bkernel
Cbias
Dtrainable_variables
E	variables
Fregularization_losses
G	keras_api
h

Hkernel
Ibias
Jtrainable_variables
K	variables
Lregularization_losses
M	keras_api
?
Niter

Obeta_1

Pbeta_2
	Qdecay
Rlearning_ratem?m?m?m?"m?#m?,m?-m?6m?7m?<m?=m?Bm?Cm?Hm?Im?v?v?v?v?"v?#v?,v?-v?6v?7v?<v?=v?Bv?Cv?Hv?Iv?
v
0
1
2
3
"4
#5
,6
-7
68
79
<10
=11
B12
C13
H14
I15
v
0
1
2
3
"4
#5
,6
-7
68
79
<10
=11
B12
C13
H14
I15
 
?
trainable_variables
	variables
Slayer_metrics
Tlayer_regularization_losses
Unon_trainable_variables
Vmetrics
regularization_losses

Wlayers
 
 
 
 
?
trainable_variables
Xlayer_metrics
Ylayer_regularization_losses
	variables
Znon_trainable_variables
[metrics
regularization_losses

\layers
ge
VARIABLE_VALUEencoder_layer_extra1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEencoder_layer_extra1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
trainable_variables
]layer_metrics
^layer_regularization_losses
	variables
_non_trainable_variables
`metrics
regularization_losses

alayers
ge
VARIABLE_VALUEencoder_layer_extra2/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEencoder_layer_extra2/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
trainable_variables
blayer_metrics
clayer_regularization_losses
	variables
dnon_trainable_variables
emetrics
 regularization_losses

flayers
b`
VARIABLE_VALUEencoder_layer_1/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUEencoder_layer_1/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

"0
#1

"0
#1
 
?
$trainable_variables
glayer_metrics
hlayer_regularization_losses
%	variables
inon_trainable_variables
jmetrics
&regularization_losses

klayers
 
 
 
?
(trainable_variables
llayer_metrics
mlayer_regularization_losses
)	variables
nnon_trainable_variables
ometrics
*regularization_losses

players
WU
VARIABLE_VALUEcode/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUE	code/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

,0
-1

,0
-1
 
?
.trainable_variables
qlayer_metrics
rlayer_regularization_losses
/	variables
snon_trainable_variables
tmetrics
0regularization_losses

ulayers
 
 
 
?
2trainable_variables
vlayer_metrics
wlayer_regularization_losses
3	variables
xnon_trainable_variables
ymetrics
4regularization_losses

zlayers
ge
VARIABLE_VALUEdecoder_layer_extra1/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEdecoder_layer_extra1/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

60
71

60
71
 
?
8trainable_variables
{layer_metrics
|layer_regularization_losses
9	variables
}non_trainable_variables
~metrics
:regularization_losses

layers
ge
VARIABLE_VALUEdecoder_layer_extra2/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEdecoder_layer_extra2/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE

<0
=1

<0
=1
 
?
>trainable_variables
?layer_metrics
 ?layer_regularization_losses
?	variables
?non_trainable_variables
?metrics
@regularization_losses
?layers
b`
VARIABLE_VALUEdecoder_layer_1/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUEdecoder_layer_1/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE

B0
C1

B0
C1
 
?
Dtrainable_variables
?layer_metrics
 ?layer_regularization_losses
E	variables
?non_trainable_variables
?metrics
Fregularization_losses
?layers
b`
VARIABLE_VALUEdecoder_layer_2/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUEdecoder_layer_2/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE

H0
I1

H0
I1
 
?
Jtrainable_variables
?layer_metrics
 ?layer_regularization_losses
K	variables
?non_trainable_variables
?metrics
Lregularization_losses
?layers
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
 
 
 

?0
?1
N
0
1
2
3
4
5
6
7
	8

9
10
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
8

?total

?count
?	variables
?	keras_api
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
??
VARIABLE_VALUE"Adam/encoder_layer_extra1/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE Adam/encoder_layer_extra1/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/encoder_layer_extra2/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE Adam/encoder_layer_extra2/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/encoder_layer_1/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUEAdam/encoder_layer_1/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/code/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/code/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/decoder_layer_extra1/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE Adam/decoder_layer_extra1/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/decoder_layer_extra2/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE Adam/decoder_layer_extra2/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/decoder_layer_1/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUEAdam/decoder_layer_1/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/decoder_layer_2/kernel/mRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUEAdam/decoder_layer_2/bias/mPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/encoder_layer_extra1/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE Adam/encoder_layer_extra1/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/encoder_layer_extra2/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE Adam/encoder_layer_extra2/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/encoder_layer_1/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUEAdam/encoder_layer_1/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/code/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/code/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/decoder_layer_extra1/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE Adam/decoder_layer_extra1/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/decoder_layer_extra2/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE Adam/decoder_layer_extra2/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/decoder_layer_1/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUEAdam/decoder_layer_1/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/decoder_layer_2/kernel/vRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUEAdam/decoder_layer_2/bias/vPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_input_1Placeholder*/
_output_shapes
:?????????*
dtype0*$
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1encoder_layer_extra1/kernelencoder_layer_extra1/biasencoder_layer_extra2/kernelencoder_layer_extra2/biasencoder_layer_1/kernelencoder_layer_1/biascode/kernel	code/biasdecoder_layer_extra1/kerneldecoder_layer_extra1/biasdecoder_layer_extra2/kerneldecoder_layer_extra2/biasdecoder_layer_1/kerneldecoder_layer_1/biasdecoder_layer_2/kerneldecoder_layer_2/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *,
f'R%
#__inference_signature_wrapper_13149
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename/encoder_layer_extra1/kernel/Read/ReadVariableOp-encoder_layer_extra1/bias/Read/ReadVariableOp/encoder_layer_extra2/kernel/Read/ReadVariableOp-encoder_layer_extra2/bias/Read/ReadVariableOp*encoder_layer_1/kernel/Read/ReadVariableOp(encoder_layer_1/bias/Read/ReadVariableOpcode/kernel/Read/ReadVariableOpcode/bias/Read/ReadVariableOp/decoder_layer_extra1/kernel/Read/ReadVariableOp-decoder_layer_extra1/bias/Read/ReadVariableOp/decoder_layer_extra2/kernel/Read/ReadVariableOp-decoder_layer_extra2/bias/Read/ReadVariableOp*decoder_layer_1/kernel/Read/ReadVariableOp(decoder_layer_1/bias/Read/ReadVariableOp*decoder_layer_2/kernel/Read/ReadVariableOp(decoder_layer_2/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp6Adam/encoder_layer_extra1/kernel/m/Read/ReadVariableOp4Adam/encoder_layer_extra1/bias/m/Read/ReadVariableOp6Adam/encoder_layer_extra2/kernel/m/Read/ReadVariableOp4Adam/encoder_layer_extra2/bias/m/Read/ReadVariableOp1Adam/encoder_layer_1/kernel/m/Read/ReadVariableOp/Adam/encoder_layer_1/bias/m/Read/ReadVariableOp&Adam/code/kernel/m/Read/ReadVariableOp$Adam/code/bias/m/Read/ReadVariableOp6Adam/decoder_layer_extra1/kernel/m/Read/ReadVariableOp4Adam/decoder_layer_extra1/bias/m/Read/ReadVariableOp6Adam/decoder_layer_extra2/kernel/m/Read/ReadVariableOp4Adam/decoder_layer_extra2/bias/m/Read/ReadVariableOp1Adam/decoder_layer_1/kernel/m/Read/ReadVariableOp/Adam/decoder_layer_1/bias/m/Read/ReadVariableOp1Adam/decoder_layer_2/kernel/m/Read/ReadVariableOp/Adam/decoder_layer_2/bias/m/Read/ReadVariableOp6Adam/encoder_layer_extra1/kernel/v/Read/ReadVariableOp4Adam/encoder_layer_extra1/bias/v/Read/ReadVariableOp6Adam/encoder_layer_extra2/kernel/v/Read/ReadVariableOp4Adam/encoder_layer_extra2/bias/v/Read/ReadVariableOp1Adam/encoder_layer_1/kernel/v/Read/ReadVariableOp/Adam/encoder_layer_1/bias/v/Read/ReadVariableOp&Adam/code/kernel/v/Read/ReadVariableOp$Adam/code/bias/v/Read/ReadVariableOp6Adam/decoder_layer_extra1/kernel/v/Read/ReadVariableOp4Adam/decoder_layer_extra1/bias/v/Read/ReadVariableOp6Adam/decoder_layer_extra2/kernel/v/Read/ReadVariableOp4Adam/decoder_layer_extra2/bias/v/Read/ReadVariableOp1Adam/decoder_layer_1/kernel/v/Read/ReadVariableOp/Adam/decoder_layer_1/bias/v/Read/ReadVariableOp1Adam/decoder_layer_2/kernel/v/Read/ReadVariableOp/Adam/decoder_layer_2/bias/v/Read/ReadVariableOpConst*F
Tin?
=2;	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *'
f"R 
__inference__traced_save_13811
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameencoder_layer_extra1/kernelencoder_layer_extra1/biasencoder_layer_extra2/kernelencoder_layer_extra2/biasencoder_layer_1/kernelencoder_layer_1/biascode/kernel	code/biasdecoder_layer_extra1/kerneldecoder_layer_extra1/biasdecoder_layer_extra2/kerneldecoder_layer_extra2/biasdecoder_layer_1/kerneldecoder_layer_1/biasdecoder_layer_2/kerneldecoder_layer_2/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1"Adam/encoder_layer_extra1/kernel/m Adam/encoder_layer_extra1/bias/m"Adam/encoder_layer_extra2/kernel/m Adam/encoder_layer_extra2/bias/mAdam/encoder_layer_1/kernel/mAdam/encoder_layer_1/bias/mAdam/code/kernel/mAdam/code/bias/m"Adam/decoder_layer_extra1/kernel/m Adam/decoder_layer_extra1/bias/m"Adam/decoder_layer_extra2/kernel/m Adam/decoder_layer_extra2/bias/mAdam/decoder_layer_1/kernel/mAdam/decoder_layer_1/bias/mAdam/decoder_layer_2/kernel/mAdam/decoder_layer_2/bias/m"Adam/encoder_layer_extra1/kernel/v Adam/encoder_layer_extra1/bias/v"Adam/encoder_layer_extra2/kernel/v Adam/encoder_layer_extra2/bias/vAdam/encoder_layer_1/kernel/vAdam/encoder_layer_1/bias/vAdam/code/kernel/vAdam/code/bias/v"Adam/decoder_layer_extra1/kernel/v Adam/decoder_layer_extra1/bias/v"Adam/decoder_layer_extra2/kernel/v Adam/decoder_layer_extra2/bias/vAdam/decoder_layer_1/kernel/vAdam/decoder_layer_1/bias/vAdam/decoder_layer_2/kernel/vAdam/decoder_layer_2/bias/v*E
Tin>
<2:*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? **
f%R#
!__inference__traced_restore_13992??
?
?
$__inference_code_layer_call_fn_13598

inputs
unknown:	?
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_code_layer_call_and_return_conditional_losses_127162
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?<
?
J__inference_autoencoder_cnn_layer_call_and_return_conditional_losses_13104
input_14
encoder_layer_extra1_13061:(
encoder_layer_extra1_13063:4
encoder_layer_extra2_13066:(
encoder_layer_extra2_13068:/
encoder_layer_1_13071:#
encoder_layer_1_13073:

code_13077:	?

code_13079:4
decoder_layer_extra1_13083:(
decoder_layer_extra1_13085:4
decoder_layer_extra2_13088:(
decoder_layer_extra2_13090:/
decoder_layer_1_13093:#
decoder_layer_1_13095:/
decoder_layer_2_13098:#
decoder_layer_2_13100:
identity??code/StatefulPartitionedCall?'decoder_layer_1/StatefulPartitionedCall?'decoder_layer_2/StatefulPartitionedCall?,decoder_layer_extra1/StatefulPartitionedCall?,decoder_layer_extra2/StatefulPartitionedCall?'encoder_layer_1/StatefulPartitionedCall?,encoder_layer_extra1/StatefulPartitionedCall?,encoder_layer_extra2/StatefulPartitionedCall?
reshape/PartitionedCallPartitionedCallinput_1*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_reshape_layer_call_and_return_conditional_losses_126452
reshape/PartitionedCall?
,encoder_layer_extra1/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0encoder_layer_extra1_13061encoder_layer_extra1_13063*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_encoder_layer_extra1_layer_call_and_return_conditional_losses_126582.
,encoder_layer_extra1/StatefulPartitionedCall?
,encoder_layer_extra2/StatefulPartitionedCallStatefulPartitionedCall5encoder_layer_extra1/StatefulPartitionedCall:output:0encoder_layer_extra2_13066encoder_layer_extra2_13068*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_encoder_layer_extra2_layer_call_and_return_conditional_losses_126752.
,encoder_layer_extra2/StatefulPartitionedCall?
'encoder_layer_1/StatefulPartitionedCallStatefulPartitionedCall5encoder_layer_extra2/StatefulPartitionedCall:output:0encoder_layer_1_13071encoder_layer_1_13073*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_encoder_layer_1_layer_call_and_return_conditional_losses_126922)
'encoder_layer_1/StatefulPartitionedCall?
flatten/PartitionedCallPartitionedCall0encoder_layer_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_127042
flatten/PartitionedCall?
code/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0
code_13077
code_13079*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_code_layer_call_and_return_conditional_losses_127162
code/StatefulPartitionedCall?
reshape_1/PartitionedCallPartitionedCall%code/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_reshape_1_layer_call_and_return_conditional_losses_127362
reshape_1/PartitionedCall?
,decoder_layer_extra1/StatefulPartitionedCallStatefulPartitionedCall"reshape_1/PartitionedCall:output:0decoder_layer_extra1_13083decoder_layer_extra1_13085*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_decoder_layer_extra1_layer_call_and_return_conditional_losses_124792.
,decoder_layer_extra1/StatefulPartitionedCall?
,decoder_layer_extra2/StatefulPartitionedCallStatefulPartitionedCall5decoder_layer_extra1/StatefulPartitionedCall:output:0decoder_layer_extra2_13088decoder_layer_extra2_13090*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_decoder_layer_extra2_layer_call_and_return_conditional_losses_125242.
,decoder_layer_extra2/StatefulPartitionedCall?
'decoder_layer_1/StatefulPartitionedCallStatefulPartitionedCall5decoder_layer_extra2/StatefulPartitionedCall:output:0decoder_layer_1_13093decoder_layer_1_13095*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_decoder_layer_1_layer_call_and_return_conditional_losses_125692)
'decoder_layer_1/StatefulPartitionedCall?
'decoder_layer_2/StatefulPartitionedCallStatefulPartitionedCall0decoder_layer_1/StatefulPartitionedCall:output:0decoder_layer_2_13098decoder_layer_2_13100*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_decoder_layer_2_layer_call_and_return_conditional_losses_126142)
'decoder_layer_2/StatefulPartitionedCall?
IdentityIdentity0decoder_layer_2/StatefulPartitionedCall:output:0^code/StatefulPartitionedCall(^decoder_layer_1/StatefulPartitionedCall(^decoder_layer_2/StatefulPartitionedCall-^decoder_layer_extra1/StatefulPartitionedCall-^decoder_layer_extra2/StatefulPartitionedCall(^encoder_layer_1/StatefulPartitionedCall-^encoder_layer_extra1/StatefulPartitionedCall-^encoder_layer_extra2/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:?????????: : : : : : : : : : : : : : : : 2<
code/StatefulPartitionedCallcode/StatefulPartitionedCall2R
'decoder_layer_1/StatefulPartitionedCall'decoder_layer_1/StatefulPartitionedCall2R
'decoder_layer_2/StatefulPartitionedCall'decoder_layer_2/StatefulPartitionedCall2\
,decoder_layer_extra1/StatefulPartitionedCall,decoder_layer_extra1/StatefulPartitionedCall2\
,decoder_layer_extra2/StatefulPartitionedCall,decoder_layer_extra2/StatefulPartitionedCall2R
'encoder_layer_1/StatefulPartitionedCall'encoder_layer_1/StatefulPartitionedCall2\
,encoder_layer_extra1/StatefulPartitionedCall,encoder_layer_extra1/StatefulPartitionedCall2\
,encoder_layer_extra2/StatefulPartitionedCall,encoder_layer_extra2/StatefulPartitionedCall:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_1
?
?
/__inference_decoder_layer_2_layer_call_fn_12624

inputs!
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_decoder_layer_2_layer_call_and_return_conditional_losses_126142
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
^
B__inference_reshape_layer_call_and_return_conditional_losses_12645

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/2d
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/3?
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapew
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:?????????2	
Reshapel
IdentityIdentityReshape:output:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?<
?
J__inference_autoencoder_cnn_layer_call_and_return_conditional_losses_13057
input_14
encoder_layer_extra1_13014:(
encoder_layer_extra1_13016:4
encoder_layer_extra2_13019:(
encoder_layer_extra2_13021:/
encoder_layer_1_13024:#
encoder_layer_1_13026:

code_13030:	?

code_13032:4
decoder_layer_extra1_13036:(
decoder_layer_extra1_13038:4
decoder_layer_extra2_13041:(
decoder_layer_extra2_13043:/
decoder_layer_1_13046:#
decoder_layer_1_13048:/
decoder_layer_2_13051:#
decoder_layer_2_13053:
identity??code/StatefulPartitionedCall?'decoder_layer_1/StatefulPartitionedCall?'decoder_layer_2/StatefulPartitionedCall?,decoder_layer_extra1/StatefulPartitionedCall?,decoder_layer_extra2/StatefulPartitionedCall?'encoder_layer_1/StatefulPartitionedCall?,encoder_layer_extra1/StatefulPartitionedCall?,encoder_layer_extra2/StatefulPartitionedCall?
reshape/PartitionedCallPartitionedCallinput_1*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_reshape_layer_call_and_return_conditional_losses_126452
reshape/PartitionedCall?
,encoder_layer_extra1/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0encoder_layer_extra1_13014encoder_layer_extra1_13016*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_encoder_layer_extra1_layer_call_and_return_conditional_losses_126582.
,encoder_layer_extra1/StatefulPartitionedCall?
,encoder_layer_extra2/StatefulPartitionedCallStatefulPartitionedCall5encoder_layer_extra1/StatefulPartitionedCall:output:0encoder_layer_extra2_13019encoder_layer_extra2_13021*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_encoder_layer_extra2_layer_call_and_return_conditional_losses_126752.
,encoder_layer_extra2/StatefulPartitionedCall?
'encoder_layer_1/StatefulPartitionedCallStatefulPartitionedCall5encoder_layer_extra2/StatefulPartitionedCall:output:0encoder_layer_1_13024encoder_layer_1_13026*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_encoder_layer_1_layer_call_and_return_conditional_losses_126922)
'encoder_layer_1/StatefulPartitionedCall?
flatten/PartitionedCallPartitionedCall0encoder_layer_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_127042
flatten/PartitionedCall?
code/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0
code_13030
code_13032*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_code_layer_call_and_return_conditional_losses_127162
code/StatefulPartitionedCall?
reshape_1/PartitionedCallPartitionedCall%code/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_reshape_1_layer_call_and_return_conditional_losses_127362
reshape_1/PartitionedCall?
,decoder_layer_extra1/StatefulPartitionedCallStatefulPartitionedCall"reshape_1/PartitionedCall:output:0decoder_layer_extra1_13036decoder_layer_extra1_13038*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_decoder_layer_extra1_layer_call_and_return_conditional_losses_124792.
,decoder_layer_extra1/StatefulPartitionedCall?
,decoder_layer_extra2/StatefulPartitionedCallStatefulPartitionedCall5decoder_layer_extra1/StatefulPartitionedCall:output:0decoder_layer_extra2_13041decoder_layer_extra2_13043*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_decoder_layer_extra2_layer_call_and_return_conditional_losses_125242.
,decoder_layer_extra2/StatefulPartitionedCall?
'decoder_layer_1/StatefulPartitionedCallStatefulPartitionedCall5decoder_layer_extra2/StatefulPartitionedCall:output:0decoder_layer_1_13046decoder_layer_1_13048*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_decoder_layer_1_layer_call_and_return_conditional_losses_125692)
'decoder_layer_1/StatefulPartitionedCall?
'decoder_layer_2/StatefulPartitionedCallStatefulPartitionedCall0decoder_layer_1/StatefulPartitionedCall:output:0decoder_layer_2_13051decoder_layer_2_13053*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_decoder_layer_2_layer_call_and_return_conditional_losses_126142)
'decoder_layer_2/StatefulPartitionedCall?
IdentityIdentity0decoder_layer_2/StatefulPartitionedCall:output:0^code/StatefulPartitionedCall(^decoder_layer_1/StatefulPartitionedCall(^decoder_layer_2/StatefulPartitionedCall-^decoder_layer_extra1/StatefulPartitionedCall-^decoder_layer_extra2/StatefulPartitionedCall(^encoder_layer_1/StatefulPartitionedCall-^encoder_layer_extra1/StatefulPartitionedCall-^encoder_layer_extra2/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:?????????: : : : : : : : : : : : : : : : 2<
code/StatefulPartitionedCallcode/StatefulPartitionedCall2R
'decoder_layer_1/StatefulPartitionedCall'decoder_layer_1/StatefulPartitionedCall2R
'decoder_layer_2/StatefulPartitionedCall'decoder_layer_2/StatefulPartitionedCall2\
,decoder_layer_extra1/StatefulPartitionedCall,decoder_layer_extra1/StatefulPartitionedCall2\
,decoder_layer_extra2/StatefulPartitionedCall,decoder_layer_extra2/StatefulPartitionedCall2R
'encoder_layer_1/StatefulPartitionedCall'encoder_layer_1/StatefulPartitionedCall2\
,encoder_layer_extra1/StatefulPartitionedCall,encoder_layer_extra1/StatefulPartitionedCall2\
,encoder_layer_extra2/StatefulPartitionedCall,encoder_layer_extra2/StatefulPartitionedCall:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_1
?
?
/__inference_autoencoder_cnn_layer_call_fn_13452

inputs!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:#
	unknown_3:
	unknown_4:
	unknown_5:	?
	unknown_6:#
	unknown_7:
	unknown_8:#
	unknown_9:

unknown_10:$

unknown_11:

unknown_12:$

unknown_13:

unknown_14:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_autoencoder_cnn_layer_call_and_return_conditional_losses_127592
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:?????????: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
4__inference_encoder_layer_extra1_layer_call_fn_13528

inputs!
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_encoder_layer_extra1_layer_call_and_return_conditional_losses_126582
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?
J__inference_autoencoder_cnn_layer_call_and_return_conditional_losses_13282

inputsM
3encoder_layer_extra1_conv2d_readvariableop_resource:B
4encoder_layer_extra1_biasadd_readvariableop_resource:M
3encoder_layer_extra2_conv2d_readvariableop_resource:B
4encoder_layer_extra2_biasadd_readvariableop_resource:H
.encoder_layer_1_conv2d_readvariableop_resource:=
/encoder_layer_1_biasadd_readvariableop_resource:6
#code_matmul_readvariableop_resource:	?2
$code_biasadd_readvariableop_resource:W
=decoder_layer_extra1_conv2d_transpose_readvariableop_resource:B
4decoder_layer_extra1_biasadd_readvariableop_resource:W
=decoder_layer_extra2_conv2d_transpose_readvariableop_resource:B
4decoder_layer_extra2_biasadd_readvariableop_resource:R
8decoder_layer_1_conv2d_transpose_readvariableop_resource:=
/decoder_layer_1_biasadd_readvariableop_resource:R
8decoder_layer_2_conv2d_transpose_readvariableop_resource:=
/decoder_layer_2_biasadd_readvariableop_resource:
identity??code/BiasAdd/ReadVariableOp?code/MatMul/ReadVariableOp?&decoder_layer_1/BiasAdd/ReadVariableOp?/decoder_layer_1/conv2d_transpose/ReadVariableOp?&decoder_layer_2/BiasAdd/ReadVariableOp?/decoder_layer_2/conv2d_transpose/ReadVariableOp?+decoder_layer_extra1/BiasAdd/ReadVariableOp?4decoder_layer_extra1/conv2d_transpose/ReadVariableOp?+decoder_layer_extra2/BiasAdd/ReadVariableOp?4decoder_layer_extra2/conv2d_transpose/ReadVariableOp?&encoder_layer_1/BiasAdd/ReadVariableOp?%encoder_layer_1/Conv2D/ReadVariableOp?+encoder_layer_extra1/BiasAdd/ReadVariableOp?*encoder_layer_extra1/Conv2D/ReadVariableOp?+encoder_layer_extra2/BiasAdd/ReadVariableOp?*encoder_layer_extra2/Conv2D/ReadVariableOpT
reshape/ShapeShapeinputs*
T0*
_output_shapes
:2
reshape/Shape?
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape/strided_slice/stack?
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_1?
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_2?
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape/strided_slicet
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/1t
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/2t
reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/3?
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0 reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape/Reshape/shape?
reshape/ReshapeReshapeinputsreshape/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????2
reshape/Reshape?
*encoder_layer_extra1/Conv2D/ReadVariableOpReadVariableOp3encoder_layer_extra1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02,
*encoder_layer_extra1/Conv2D/ReadVariableOp?
encoder_layer_extra1/Conv2DConv2Dreshape/Reshape:output:02encoder_layer_extra1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
encoder_layer_extra1/Conv2D?
+encoder_layer_extra1/BiasAdd/ReadVariableOpReadVariableOp4encoder_layer_extra1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+encoder_layer_extra1/BiasAdd/ReadVariableOp?
encoder_layer_extra1/BiasAddBiasAdd$encoder_layer_extra1/Conv2D:output:03encoder_layer_extra1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
encoder_layer_extra1/BiasAdd?
encoder_layer_extra1/ReluRelu%encoder_layer_extra1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
encoder_layer_extra1/Relu?
*encoder_layer_extra2/Conv2D/ReadVariableOpReadVariableOp3encoder_layer_extra2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02,
*encoder_layer_extra2/Conv2D/ReadVariableOp?
encoder_layer_extra2/Conv2DConv2D'encoder_layer_extra1/Relu:activations:02encoder_layer_extra2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
encoder_layer_extra2/Conv2D?
+encoder_layer_extra2/BiasAdd/ReadVariableOpReadVariableOp4encoder_layer_extra2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+encoder_layer_extra2/BiasAdd/ReadVariableOp?
encoder_layer_extra2/BiasAddBiasAdd$encoder_layer_extra2/Conv2D:output:03encoder_layer_extra2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
encoder_layer_extra2/BiasAdd?
encoder_layer_extra2/ReluRelu%encoder_layer_extra2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
encoder_layer_extra2/Relu?
%encoder_layer_1/Conv2D/ReadVariableOpReadVariableOp.encoder_layer_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02'
%encoder_layer_1/Conv2D/ReadVariableOp?
encoder_layer_1/Conv2DConv2D'encoder_layer_extra2/Relu:activations:0-encoder_layer_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
encoder_layer_1/Conv2D?
&encoder_layer_1/BiasAdd/ReadVariableOpReadVariableOp/encoder_layer_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&encoder_layer_1/BiasAdd/ReadVariableOp?
encoder_layer_1/BiasAddBiasAddencoder_layer_1/Conv2D:output:0.encoder_layer_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
encoder_layer_1/BiasAdd?
encoder_layer_1/ReluRelu encoder_layer_1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
encoder_layer_1/Reluo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten/Const?
flatten/ReshapeReshape"encoder_layer_1/Relu:activations:0flatten/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten/Reshape?
code/MatMul/ReadVariableOpReadVariableOp#code_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
code/MatMul/ReadVariableOp?
code/MatMulMatMulflatten/Reshape:output:0"code/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
code/MatMul?
code/BiasAdd/ReadVariableOpReadVariableOp$code_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
code/BiasAdd/ReadVariableOp?
code/BiasAddBiasAddcode/MatMul:product:0#code/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
code/BiasAddg
reshape_1/ShapeShapecode/BiasAdd:output:0*
T0*
_output_shapes
:2
reshape_1/Shape?
reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape_1/strided_slice/stack?
reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_1/strided_slice/stack_1?
reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_1/strided_slice/stack_2?
reshape_1/strided_sliceStridedSlicereshape_1/Shape:output:0&reshape_1/strided_slice/stack:output:0(reshape_1/strided_slice/stack_1:output:0(reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_1/strided_slicex
reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_1/Reshape/shape/1x
reshape_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_1/Reshape/shape/2x
reshape_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_1/Reshape/shape/3?
reshape_1/Reshape/shapePack reshape_1/strided_slice:output:0"reshape_1/Reshape/shape/1:output:0"reshape_1/Reshape/shape/2:output:0"reshape_1/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape_1/Reshape/shape?
reshape_1/ReshapeReshapecode/BiasAdd:output:0 reshape_1/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????2
reshape_1/Reshape?
decoder_layer_extra1/ShapeShapereshape_1/Reshape:output:0*
T0*
_output_shapes
:2
decoder_layer_extra1/Shape?
(decoder_layer_extra1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(decoder_layer_extra1/strided_slice/stack?
*decoder_layer_extra1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*decoder_layer_extra1/strided_slice/stack_1?
*decoder_layer_extra1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*decoder_layer_extra1/strided_slice/stack_2?
"decoder_layer_extra1/strided_sliceStridedSlice#decoder_layer_extra1/Shape:output:01decoder_layer_extra1/strided_slice/stack:output:03decoder_layer_extra1/strided_slice/stack_1:output:03decoder_layer_extra1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"decoder_layer_extra1/strided_slice~
decoder_layer_extra1/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2
decoder_layer_extra1/stack/1~
decoder_layer_extra1/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
decoder_layer_extra1/stack/2~
decoder_layer_extra1/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
decoder_layer_extra1/stack/3?
decoder_layer_extra1/stackPack+decoder_layer_extra1/strided_slice:output:0%decoder_layer_extra1/stack/1:output:0%decoder_layer_extra1/stack/2:output:0%decoder_layer_extra1/stack/3:output:0*
N*
T0*
_output_shapes
:2
decoder_layer_extra1/stack?
*decoder_layer_extra1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*decoder_layer_extra1/strided_slice_1/stack?
,decoder_layer_extra1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,decoder_layer_extra1/strided_slice_1/stack_1?
,decoder_layer_extra1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,decoder_layer_extra1/strided_slice_1/stack_2?
$decoder_layer_extra1/strided_slice_1StridedSlice#decoder_layer_extra1/stack:output:03decoder_layer_extra1/strided_slice_1/stack:output:05decoder_layer_extra1/strided_slice_1/stack_1:output:05decoder_layer_extra1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2&
$decoder_layer_extra1/strided_slice_1?
4decoder_layer_extra1/conv2d_transpose/ReadVariableOpReadVariableOp=decoder_layer_extra1_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype026
4decoder_layer_extra1/conv2d_transpose/ReadVariableOp?
%decoder_layer_extra1/conv2d_transposeConv2DBackpropInput#decoder_layer_extra1/stack:output:0<decoder_layer_extra1/conv2d_transpose/ReadVariableOp:value:0reshape_1/Reshape:output:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2'
%decoder_layer_extra1/conv2d_transpose?
+decoder_layer_extra1/BiasAdd/ReadVariableOpReadVariableOp4decoder_layer_extra1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+decoder_layer_extra1/BiasAdd/ReadVariableOp?
decoder_layer_extra1/BiasAddBiasAdd.decoder_layer_extra1/conv2d_transpose:output:03decoder_layer_extra1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
decoder_layer_extra1/BiasAdd?
decoder_layer_extra1/ReluRelu%decoder_layer_extra1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
decoder_layer_extra1/Relu?
decoder_layer_extra2/ShapeShape'decoder_layer_extra1/Relu:activations:0*
T0*
_output_shapes
:2
decoder_layer_extra2/Shape?
(decoder_layer_extra2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(decoder_layer_extra2/strided_slice/stack?
*decoder_layer_extra2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*decoder_layer_extra2/strided_slice/stack_1?
*decoder_layer_extra2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*decoder_layer_extra2/strided_slice/stack_2?
"decoder_layer_extra2/strided_sliceStridedSlice#decoder_layer_extra2/Shape:output:01decoder_layer_extra2/strided_slice/stack:output:03decoder_layer_extra2/strided_slice/stack_1:output:03decoder_layer_extra2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"decoder_layer_extra2/strided_slice~
decoder_layer_extra2/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2
decoder_layer_extra2/stack/1~
decoder_layer_extra2/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
decoder_layer_extra2/stack/2~
decoder_layer_extra2/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
decoder_layer_extra2/stack/3?
decoder_layer_extra2/stackPack+decoder_layer_extra2/strided_slice:output:0%decoder_layer_extra2/stack/1:output:0%decoder_layer_extra2/stack/2:output:0%decoder_layer_extra2/stack/3:output:0*
N*
T0*
_output_shapes
:2
decoder_layer_extra2/stack?
*decoder_layer_extra2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*decoder_layer_extra2/strided_slice_1/stack?
,decoder_layer_extra2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,decoder_layer_extra2/strided_slice_1/stack_1?
,decoder_layer_extra2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,decoder_layer_extra2/strided_slice_1/stack_2?
$decoder_layer_extra2/strided_slice_1StridedSlice#decoder_layer_extra2/stack:output:03decoder_layer_extra2/strided_slice_1/stack:output:05decoder_layer_extra2/strided_slice_1/stack_1:output:05decoder_layer_extra2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2&
$decoder_layer_extra2/strided_slice_1?
4decoder_layer_extra2/conv2d_transpose/ReadVariableOpReadVariableOp=decoder_layer_extra2_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype026
4decoder_layer_extra2/conv2d_transpose/ReadVariableOp?
%decoder_layer_extra2/conv2d_transposeConv2DBackpropInput#decoder_layer_extra2/stack:output:0<decoder_layer_extra2/conv2d_transpose/ReadVariableOp:value:0'decoder_layer_extra1/Relu:activations:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2'
%decoder_layer_extra2/conv2d_transpose?
+decoder_layer_extra2/BiasAdd/ReadVariableOpReadVariableOp4decoder_layer_extra2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+decoder_layer_extra2/BiasAdd/ReadVariableOp?
decoder_layer_extra2/BiasAddBiasAdd.decoder_layer_extra2/conv2d_transpose:output:03decoder_layer_extra2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
decoder_layer_extra2/BiasAdd?
decoder_layer_extra2/ReluRelu%decoder_layer_extra2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
decoder_layer_extra2/Relu?
decoder_layer_1/ShapeShape'decoder_layer_extra2/Relu:activations:0*
T0*
_output_shapes
:2
decoder_layer_1/Shape?
#decoder_layer_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2%
#decoder_layer_1/strided_slice/stack?
%decoder_layer_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%decoder_layer_1/strided_slice/stack_1?
%decoder_layer_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%decoder_layer_1/strided_slice/stack_2?
decoder_layer_1/strided_sliceStridedSlicedecoder_layer_1/Shape:output:0,decoder_layer_1/strided_slice/stack:output:0.decoder_layer_1/strided_slice/stack_1:output:0.decoder_layer_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
decoder_layer_1/strided_slicet
decoder_layer_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2
decoder_layer_1/stack/1t
decoder_layer_1/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
decoder_layer_1/stack/2t
decoder_layer_1/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
decoder_layer_1/stack/3?
decoder_layer_1/stackPack&decoder_layer_1/strided_slice:output:0 decoder_layer_1/stack/1:output:0 decoder_layer_1/stack/2:output:0 decoder_layer_1/stack/3:output:0*
N*
T0*
_output_shapes
:2
decoder_layer_1/stack?
%decoder_layer_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%decoder_layer_1/strided_slice_1/stack?
'decoder_layer_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'decoder_layer_1/strided_slice_1/stack_1?
'decoder_layer_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'decoder_layer_1/strided_slice_1/stack_2?
decoder_layer_1/strided_slice_1StridedSlicedecoder_layer_1/stack:output:0.decoder_layer_1/strided_slice_1/stack:output:00decoder_layer_1/strided_slice_1/stack_1:output:00decoder_layer_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
decoder_layer_1/strided_slice_1?
/decoder_layer_1/conv2d_transpose/ReadVariableOpReadVariableOp8decoder_layer_1_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype021
/decoder_layer_1/conv2d_transpose/ReadVariableOp?
 decoder_layer_1/conv2d_transposeConv2DBackpropInputdecoder_layer_1/stack:output:07decoder_layer_1/conv2d_transpose/ReadVariableOp:value:0'decoder_layer_extra2/Relu:activations:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2"
 decoder_layer_1/conv2d_transpose?
&decoder_layer_1/BiasAdd/ReadVariableOpReadVariableOp/decoder_layer_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&decoder_layer_1/BiasAdd/ReadVariableOp?
decoder_layer_1/BiasAddBiasAdd)decoder_layer_1/conv2d_transpose:output:0.decoder_layer_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
decoder_layer_1/BiasAdd?
decoder_layer_1/ReluRelu decoder_layer_1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
decoder_layer_1/Relu?
decoder_layer_2/ShapeShape"decoder_layer_1/Relu:activations:0*
T0*
_output_shapes
:2
decoder_layer_2/Shape?
#decoder_layer_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2%
#decoder_layer_2/strided_slice/stack?
%decoder_layer_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%decoder_layer_2/strided_slice/stack_1?
%decoder_layer_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%decoder_layer_2/strided_slice/stack_2?
decoder_layer_2/strided_sliceStridedSlicedecoder_layer_2/Shape:output:0,decoder_layer_2/strided_slice/stack:output:0.decoder_layer_2/strided_slice/stack_1:output:0.decoder_layer_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
decoder_layer_2/strided_slicet
decoder_layer_2/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2
decoder_layer_2/stack/1t
decoder_layer_2/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
decoder_layer_2/stack/2t
decoder_layer_2/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
decoder_layer_2/stack/3?
decoder_layer_2/stackPack&decoder_layer_2/strided_slice:output:0 decoder_layer_2/stack/1:output:0 decoder_layer_2/stack/2:output:0 decoder_layer_2/stack/3:output:0*
N*
T0*
_output_shapes
:2
decoder_layer_2/stack?
%decoder_layer_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%decoder_layer_2/strided_slice_1/stack?
'decoder_layer_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'decoder_layer_2/strided_slice_1/stack_1?
'decoder_layer_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'decoder_layer_2/strided_slice_1/stack_2?
decoder_layer_2/strided_slice_1StridedSlicedecoder_layer_2/stack:output:0.decoder_layer_2/strided_slice_1/stack:output:00decoder_layer_2/strided_slice_1/stack_1:output:00decoder_layer_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
decoder_layer_2/strided_slice_1?
/decoder_layer_2/conv2d_transpose/ReadVariableOpReadVariableOp8decoder_layer_2_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype021
/decoder_layer_2/conv2d_transpose/ReadVariableOp?
 decoder_layer_2/conv2d_transposeConv2DBackpropInputdecoder_layer_2/stack:output:07decoder_layer_2/conv2d_transpose/ReadVariableOp:value:0"decoder_layer_1/Relu:activations:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2"
 decoder_layer_2/conv2d_transpose?
&decoder_layer_2/BiasAdd/ReadVariableOpReadVariableOp/decoder_layer_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&decoder_layer_2/BiasAdd/ReadVariableOp?
decoder_layer_2/BiasAddBiasAdd)decoder_layer_2/conv2d_transpose:output:0.decoder_layer_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
decoder_layer_2/BiasAdd?
decoder_layer_2/ReluRelu decoder_layer_2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
decoder_layer_2/Relu?
IdentityIdentity"decoder_layer_2/Relu:activations:0^code/BiasAdd/ReadVariableOp^code/MatMul/ReadVariableOp'^decoder_layer_1/BiasAdd/ReadVariableOp0^decoder_layer_1/conv2d_transpose/ReadVariableOp'^decoder_layer_2/BiasAdd/ReadVariableOp0^decoder_layer_2/conv2d_transpose/ReadVariableOp,^decoder_layer_extra1/BiasAdd/ReadVariableOp5^decoder_layer_extra1/conv2d_transpose/ReadVariableOp,^decoder_layer_extra2/BiasAdd/ReadVariableOp5^decoder_layer_extra2/conv2d_transpose/ReadVariableOp'^encoder_layer_1/BiasAdd/ReadVariableOp&^encoder_layer_1/Conv2D/ReadVariableOp,^encoder_layer_extra1/BiasAdd/ReadVariableOp+^encoder_layer_extra1/Conv2D/ReadVariableOp,^encoder_layer_extra2/BiasAdd/ReadVariableOp+^encoder_layer_extra2/Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:?????????: : : : : : : : : : : : : : : : 2:
code/BiasAdd/ReadVariableOpcode/BiasAdd/ReadVariableOp28
code/MatMul/ReadVariableOpcode/MatMul/ReadVariableOp2P
&decoder_layer_1/BiasAdd/ReadVariableOp&decoder_layer_1/BiasAdd/ReadVariableOp2b
/decoder_layer_1/conv2d_transpose/ReadVariableOp/decoder_layer_1/conv2d_transpose/ReadVariableOp2P
&decoder_layer_2/BiasAdd/ReadVariableOp&decoder_layer_2/BiasAdd/ReadVariableOp2b
/decoder_layer_2/conv2d_transpose/ReadVariableOp/decoder_layer_2/conv2d_transpose/ReadVariableOp2Z
+decoder_layer_extra1/BiasAdd/ReadVariableOp+decoder_layer_extra1/BiasAdd/ReadVariableOp2l
4decoder_layer_extra1/conv2d_transpose/ReadVariableOp4decoder_layer_extra1/conv2d_transpose/ReadVariableOp2Z
+decoder_layer_extra2/BiasAdd/ReadVariableOp+decoder_layer_extra2/BiasAdd/ReadVariableOp2l
4decoder_layer_extra2/conv2d_transpose/ReadVariableOp4decoder_layer_extra2/conv2d_transpose/ReadVariableOp2P
&encoder_layer_1/BiasAdd/ReadVariableOp&encoder_layer_1/BiasAdd/ReadVariableOp2N
%encoder_layer_1/Conv2D/ReadVariableOp%encoder_layer_1/Conv2D/ReadVariableOp2Z
+encoder_layer_extra1/BiasAdd/ReadVariableOp+encoder_layer_extra1/BiasAdd/ReadVariableOp2X
*encoder_layer_extra1/Conv2D/ReadVariableOp*encoder_layer_extra1/Conv2D/ReadVariableOp2Z
+encoder_layer_extra2/BiasAdd/ReadVariableOp+encoder_layer_extra2/BiasAdd/ReadVariableOp2X
*encoder_layer_extra2/Conv2D/ReadVariableOp*encoder_layer_extra2/Conv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
J__inference_encoder_layer_1_layer_call_and_return_conditional_losses_13559

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
/__inference_autoencoder_cnn_layer_call_fn_12794
input_1!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:#
	unknown_3:
	unknown_4:
	unknown_5:	?
	unknown_6:#
	unknown_7:
	unknown_8:#
	unknown_9:

unknown_10:$

unknown_11:

unknown_12:$

unknown_13:

unknown_14:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_autoencoder_cnn_layer_call_and_return_conditional_losses_127592
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:?????????: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_1
?
?
O__inference_encoder_layer_extra1_layer_call_and_return_conditional_losses_12658

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?<
?
J__inference_autoencoder_cnn_layer_call_and_return_conditional_losses_12938

inputs4
encoder_layer_extra1_12895:(
encoder_layer_extra1_12897:4
encoder_layer_extra2_12900:(
encoder_layer_extra2_12902:/
encoder_layer_1_12905:#
encoder_layer_1_12907:

code_12911:	?

code_12913:4
decoder_layer_extra1_12917:(
decoder_layer_extra1_12919:4
decoder_layer_extra2_12922:(
decoder_layer_extra2_12924:/
decoder_layer_1_12927:#
decoder_layer_1_12929:/
decoder_layer_2_12932:#
decoder_layer_2_12934:
identity??code/StatefulPartitionedCall?'decoder_layer_1/StatefulPartitionedCall?'decoder_layer_2/StatefulPartitionedCall?,decoder_layer_extra1/StatefulPartitionedCall?,decoder_layer_extra2/StatefulPartitionedCall?'encoder_layer_1/StatefulPartitionedCall?,encoder_layer_extra1/StatefulPartitionedCall?,encoder_layer_extra2/StatefulPartitionedCall?
reshape/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_reshape_layer_call_and_return_conditional_losses_126452
reshape/PartitionedCall?
,encoder_layer_extra1/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0encoder_layer_extra1_12895encoder_layer_extra1_12897*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_encoder_layer_extra1_layer_call_and_return_conditional_losses_126582.
,encoder_layer_extra1/StatefulPartitionedCall?
,encoder_layer_extra2/StatefulPartitionedCallStatefulPartitionedCall5encoder_layer_extra1/StatefulPartitionedCall:output:0encoder_layer_extra2_12900encoder_layer_extra2_12902*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_encoder_layer_extra2_layer_call_and_return_conditional_losses_126752.
,encoder_layer_extra2/StatefulPartitionedCall?
'encoder_layer_1/StatefulPartitionedCallStatefulPartitionedCall5encoder_layer_extra2/StatefulPartitionedCall:output:0encoder_layer_1_12905encoder_layer_1_12907*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_encoder_layer_1_layer_call_and_return_conditional_losses_126922)
'encoder_layer_1/StatefulPartitionedCall?
flatten/PartitionedCallPartitionedCall0encoder_layer_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_127042
flatten/PartitionedCall?
code/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0
code_12911
code_12913*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_code_layer_call_and_return_conditional_losses_127162
code/StatefulPartitionedCall?
reshape_1/PartitionedCallPartitionedCall%code/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_reshape_1_layer_call_and_return_conditional_losses_127362
reshape_1/PartitionedCall?
,decoder_layer_extra1/StatefulPartitionedCallStatefulPartitionedCall"reshape_1/PartitionedCall:output:0decoder_layer_extra1_12917decoder_layer_extra1_12919*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_decoder_layer_extra1_layer_call_and_return_conditional_losses_124792.
,decoder_layer_extra1/StatefulPartitionedCall?
,decoder_layer_extra2/StatefulPartitionedCallStatefulPartitionedCall5decoder_layer_extra1/StatefulPartitionedCall:output:0decoder_layer_extra2_12922decoder_layer_extra2_12924*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_decoder_layer_extra2_layer_call_and_return_conditional_losses_125242.
,decoder_layer_extra2/StatefulPartitionedCall?
'decoder_layer_1/StatefulPartitionedCallStatefulPartitionedCall5decoder_layer_extra2/StatefulPartitionedCall:output:0decoder_layer_1_12927decoder_layer_1_12929*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_decoder_layer_1_layer_call_and_return_conditional_losses_125692)
'decoder_layer_1/StatefulPartitionedCall?
'decoder_layer_2/StatefulPartitionedCallStatefulPartitionedCall0decoder_layer_1/StatefulPartitionedCall:output:0decoder_layer_2_12932decoder_layer_2_12934*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_decoder_layer_2_layer_call_and_return_conditional_losses_126142)
'decoder_layer_2/StatefulPartitionedCall?
IdentityIdentity0decoder_layer_2/StatefulPartitionedCall:output:0^code/StatefulPartitionedCall(^decoder_layer_1/StatefulPartitionedCall(^decoder_layer_2/StatefulPartitionedCall-^decoder_layer_extra1/StatefulPartitionedCall-^decoder_layer_extra2/StatefulPartitionedCall(^encoder_layer_1/StatefulPartitionedCall-^encoder_layer_extra1/StatefulPartitionedCall-^encoder_layer_extra2/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:?????????: : : : : : : : : : : : : : : : 2<
code/StatefulPartitionedCallcode/StatefulPartitionedCall2R
'decoder_layer_1/StatefulPartitionedCall'decoder_layer_1/StatefulPartitionedCall2R
'decoder_layer_2/StatefulPartitionedCall'decoder_layer_2/StatefulPartitionedCall2\
,decoder_layer_extra1/StatefulPartitionedCall,decoder_layer_extra1/StatefulPartitionedCall2\
,decoder_layer_extra2/StatefulPartitionedCall,decoder_layer_extra2/StatefulPartitionedCall2R
'encoder_layer_1/StatefulPartitionedCall'encoder_layer_1/StatefulPartitionedCall2\
,encoder_layer_extra1/StatefulPartitionedCall,encoder_layer_extra1/StatefulPartitionedCall2\
,encoder_layer_extra2/StatefulPartitionedCall,encoder_layer_extra2/StatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?&
!__inference__traced_restore_13992
file_prefixF
,assignvariableop_encoder_layer_extra1_kernel::
,assignvariableop_1_encoder_layer_extra1_bias:H
.assignvariableop_2_encoder_layer_extra2_kernel::
,assignvariableop_3_encoder_layer_extra2_bias:C
)assignvariableop_4_encoder_layer_1_kernel:5
'assignvariableop_5_encoder_layer_1_bias:1
assignvariableop_6_code_kernel:	?*
assignvariableop_7_code_bias:H
.assignvariableop_8_decoder_layer_extra1_kernel::
,assignvariableop_9_decoder_layer_extra1_bias:I
/assignvariableop_10_decoder_layer_extra2_kernel:;
-assignvariableop_11_decoder_layer_extra2_bias:D
*assignvariableop_12_decoder_layer_1_kernel:6
(assignvariableop_13_decoder_layer_1_bias:D
*assignvariableop_14_decoder_layer_2_kernel:6
(assignvariableop_15_decoder_layer_2_bias:'
assignvariableop_16_adam_iter:	 )
assignvariableop_17_adam_beta_1: )
assignvariableop_18_adam_beta_2: (
assignvariableop_19_adam_decay: 0
&assignvariableop_20_adam_learning_rate: #
assignvariableop_21_total: #
assignvariableop_22_count: %
assignvariableop_23_total_1: %
assignvariableop_24_count_1: P
6assignvariableop_25_adam_encoder_layer_extra1_kernel_m:B
4assignvariableop_26_adam_encoder_layer_extra1_bias_m:P
6assignvariableop_27_adam_encoder_layer_extra2_kernel_m:B
4assignvariableop_28_adam_encoder_layer_extra2_bias_m:K
1assignvariableop_29_adam_encoder_layer_1_kernel_m:=
/assignvariableop_30_adam_encoder_layer_1_bias_m:9
&assignvariableop_31_adam_code_kernel_m:	?2
$assignvariableop_32_adam_code_bias_m:P
6assignvariableop_33_adam_decoder_layer_extra1_kernel_m:B
4assignvariableop_34_adam_decoder_layer_extra1_bias_m:P
6assignvariableop_35_adam_decoder_layer_extra2_kernel_m:B
4assignvariableop_36_adam_decoder_layer_extra2_bias_m:K
1assignvariableop_37_adam_decoder_layer_1_kernel_m:=
/assignvariableop_38_adam_decoder_layer_1_bias_m:K
1assignvariableop_39_adam_decoder_layer_2_kernel_m:=
/assignvariableop_40_adam_decoder_layer_2_bias_m:P
6assignvariableop_41_adam_encoder_layer_extra1_kernel_v:B
4assignvariableop_42_adam_encoder_layer_extra1_bias_v:P
6assignvariableop_43_adam_encoder_layer_extra2_kernel_v:B
4assignvariableop_44_adam_encoder_layer_extra2_bias_v:K
1assignvariableop_45_adam_encoder_layer_1_kernel_v:=
/assignvariableop_46_adam_encoder_layer_1_bias_v:9
&assignvariableop_47_adam_code_kernel_v:	?2
$assignvariableop_48_adam_code_bias_v:P
6assignvariableop_49_adam_decoder_layer_extra1_kernel_v:B
4assignvariableop_50_adam_decoder_layer_extra1_bias_v:P
6assignvariableop_51_adam_decoder_layer_extra2_kernel_v:B
4assignvariableop_52_adam_decoder_layer_extra2_bias_v:K
1assignvariableop_53_adam_decoder_layer_1_kernel_v:=
/assignvariableop_54_adam_decoder_layer_1_bias_v:K
1assignvariableop_55_adam_decoder_layer_2_kernel_v:=
/assignvariableop_56_adam_decoder_layer_2_bias_v:
identity_58??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_49?AssignVariableOp_5?AssignVariableOp_50?AssignVariableOp_51?AssignVariableOp_52?AssignVariableOp_53?AssignVariableOp_54?AssignVariableOp_55?AssignVariableOp_56?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9? 
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
::*
dtype0*?
value?B?:B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
::*
dtype0*?
value~B|:B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*H
dtypes>
<2:	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp,assignvariableop_encoder_layer_extra1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp,assignvariableop_1_encoder_layer_extra1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp.assignvariableop_2_encoder_layer_extra2_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp,assignvariableop_3_encoder_layer_extra2_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp)assignvariableop_4_encoder_layer_1_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp'assignvariableop_5_encoder_layer_1_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOpassignvariableop_6_code_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOpassignvariableop_7_code_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp.assignvariableop_8_decoder_layer_extra1_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp,assignvariableop_9_decoder_layer_extra1_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp/assignvariableop_10_decoder_layer_extra2_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp-assignvariableop_11_decoder_layer_extra2_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp*assignvariableop_12_decoder_layer_1_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp(assignvariableop_13_decoder_layer_1_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp*assignvariableop_14_decoder_layer_2_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp(assignvariableop_15_decoder_layer_2_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOpassignvariableop_16_adam_iterIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOpassignvariableop_17_adam_beta_1Identity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOpassignvariableop_18_adam_beta_2Identity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOpassignvariableop_19_adam_decayIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp&assignvariableop_20_adam_learning_rateIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOpassignvariableop_21_totalIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOpassignvariableop_22_countIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOpassignvariableop_23_total_1Identity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOpassignvariableop_24_count_1Identity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp6assignvariableop_25_adam_encoder_layer_extra1_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp4assignvariableop_26_adam_encoder_layer_extra1_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp6assignvariableop_27_adam_encoder_layer_extra2_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp4assignvariableop_28_adam_encoder_layer_extra2_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp1assignvariableop_29_adam_encoder_layer_1_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp/assignvariableop_30_adam_encoder_layer_1_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp&assignvariableop_31_adam_code_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp$assignvariableop_32_adam_code_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp6assignvariableop_33_adam_decoder_layer_extra1_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp4assignvariableop_34_adam_decoder_layer_extra1_bias_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOp6assignvariableop_35_adam_decoder_layer_extra2_kernel_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOp4assignvariableop_36_adam_decoder_layer_extra2_bias_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOp1assignvariableop_37_adam_decoder_layer_1_kernel_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOp/assignvariableop_38_adam_decoder_layer_1_bias_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOp1assignvariableop_39_adam_decoder_layer_2_kernel_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOp/assignvariableop_40_adam_decoder_layer_2_bias_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41?
AssignVariableOp_41AssignVariableOp6assignvariableop_41_adam_encoder_layer_extra1_kernel_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42?
AssignVariableOp_42AssignVariableOp4assignvariableop_42_adam_encoder_layer_extra1_bias_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43?
AssignVariableOp_43AssignVariableOp6assignvariableop_43_adam_encoder_layer_extra2_kernel_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44?
AssignVariableOp_44AssignVariableOp4assignvariableop_44_adam_encoder_layer_extra2_bias_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45?
AssignVariableOp_45AssignVariableOp1assignvariableop_45_adam_encoder_layer_1_kernel_vIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46?
AssignVariableOp_46AssignVariableOp/assignvariableop_46_adam_encoder_layer_1_bias_vIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47?
AssignVariableOp_47AssignVariableOp&assignvariableop_47_adam_code_kernel_vIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48?
AssignVariableOp_48AssignVariableOp$assignvariableop_48_adam_code_bias_vIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49?
AssignVariableOp_49AssignVariableOp6assignvariableop_49_adam_decoder_layer_extra1_kernel_vIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50?
AssignVariableOp_50AssignVariableOp4assignvariableop_50_adam_decoder_layer_extra1_bias_vIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51?
AssignVariableOp_51AssignVariableOp6assignvariableop_51_adam_decoder_layer_extra2_kernel_vIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52?
AssignVariableOp_52AssignVariableOp4assignvariableop_52_adam_decoder_layer_extra2_bias_vIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53?
AssignVariableOp_53AssignVariableOp1assignvariableop_53_adam_decoder_layer_1_kernel_vIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54?
AssignVariableOp_54AssignVariableOp/assignvariableop_54_adam_decoder_layer_1_bias_vIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55?
AssignVariableOp_55AssignVariableOp1assignvariableop_55_adam_decoder_layer_2_kernel_vIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56?
AssignVariableOp_56AssignVariableOp/assignvariableop_56_adam_decoder_layer_2_bias_vIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_569
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?

Identity_57Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_57?

Identity_58IdentityIdentity_57:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_58"#
identity_58Identity_58:output:0*?
_input_shapesv
t: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
`
D__inference_reshape_1_layer_call_and_return_conditional_losses_13612

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/2d
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/3?
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapew
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:?????????2	
Reshapel
IdentityIdentityReshape:output:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs

?
 __inference__wrapped_model_12444
input_1]
Cautoencoder_cnn_encoder_layer_extra1_conv2d_readvariableop_resource:R
Dautoencoder_cnn_encoder_layer_extra1_biasadd_readvariableop_resource:]
Cautoencoder_cnn_encoder_layer_extra2_conv2d_readvariableop_resource:R
Dautoencoder_cnn_encoder_layer_extra2_biasadd_readvariableop_resource:X
>autoencoder_cnn_encoder_layer_1_conv2d_readvariableop_resource:M
?autoencoder_cnn_encoder_layer_1_biasadd_readvariableop_resource:F
3autoencoder_cnn_code_matmul_readvariableop_resource:	?B
4autoencoder_cnn_code_biasadd_readvariableop_resource:g
Mautoencoder_cnn_decoder_layer_extra1_conv2d_transpose_readvariableop_resource:R
Dautoencoder_cnn_decoder_layer_extra1_biasadd_readvariableop_resource:g
Mautoencoder_cnn_decoder_layer_extra2_conv2d_transpose_readvariableop_resource:R
Dautoencoder_cnn_decoder_layer_extra2_biasadd_readvariableop_resource:b
Hautoencoder_cnn_decoder_layer_1_conv2d_transpose_readvariableop_resource:M
?autoencoder_cnn_decoder_layer_1_biasadd_readvariableop_resource:b
Hautoencoder_cnn_decoder_layer_2_conv2d_transpose_readvariableop_resource:M
?autoencoder_cnn_decoder_layer_2_biasadd_readvariableop_resource:
identity??+autoencoder_cnn/code/BiasAdd/ReadVariableOp?*autoencoder_cnn/code/MatMul/ReadVariableOp?6autoencoder_cnn/decoder_layer_1/BiasAdd/ReadVariableOp??autoencoder_cnn/decoder_layer_1/conv2d_transpose/ReadVariableOp?6autoencoder_cnn/decoder_layer_2/BiasAdd/ReadVariableOp??autoencoder_cnn/decoder_layer_2/conv2d_transpose/ReadVariableOp?;autoencoder_cnn/decoder_layer_extra1/BiasAdd/ReadVariableOp?Dautoencoder_cnn/decoder_layer_extra1/conv2d_transpose/ReadVariableOp?;autoencoder_cnn/decoder_layer_extra2/BiasAdd/ReadVariableOp?Dautoencoder_cnn/decoder_layer_extra2/conv2d_transpose/ReadVariableOp?6autoencoder_cnn/encoder_layer_1/BiasAdd/ReadVariableOp?5autoencoder_cnn/encoder_layer_1/Conv2D/ReadVariableOp?;autoencoder_cnn/encoder_layer_extra1/BiasAdd/ReadVariableOp?:autoencoder_cnn/encoder_layer_extra1/Conv2D/ReadVariableOp?;autoencoder_cnn/encoder_layer_extra2/BiasAdd/ReadVariableOp?:autoencoder_cnn/encoder_layer_extra2/Conv2D/ReadVariableOpu
autoencoder_cnn/reshape/ShapeShapeinput_1*
T0*
_output_shapes
:2
autoencoder_cnn/reshape/Shape?
+autoencoder_cnn/reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+autoencoder_cnn/reshape/strided_slice/stack?
-autoencoder_cnn/reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-autoencoder_cnn/reshape/strided_slice/stack_1?
-autoencoder_cnn/reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-autoencoder_cnn/reshape/strided_slice/stack_2?
%autoencoder_cnn/reshape/strided_sliceStridedSlice&autoencoder_cnn/reshape/Shape:output:04autoencoder_cnn/reshape/strided_slice/stack:output:06autoencoder_cnn/reshape/strided_slice/stack_1:output:06autoencoder_cnn/reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%autoencoder_cnn/reshape/strided_slice?
'autoencoder_cnn/reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2)
'autoencoder_cnn/reshape/Reshape/shape/1?
'autoencoder_cnn/reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2)
'autoencoder_cnn/reshape/Reshape/shape/2?
'autoencoder_cnn/reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2)
'autoencoder_cnn/reshape/Reshape/shape/3?
%autoencoder_cnn/reshape/Reshape/shapePack.autoencoder_cnn/reshape/strided_slice:output:00autoencoder_cnn/reshape/Reshape/shape/1:output:00autoencoder_cnn/reshape/Reshape/shape/2:output:00autoencoder_cnn/reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2'
%autoencoder_cnn/reshape/Reshape/shape?
autoencoder_cnn/reshape/ReshapeReshapeinput_1.autoencoder_cnn/reshape/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????2!
autoencoder_cnn/reshape/Reshape?
:autoencoder_cnn/encoder_layer_extra1/Conv2D/ReadVariableOpReadVariableOpCautoencoder_cnn_encoder_layer_extra1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02<
:autoencoder_cnn/encoder_layer_extra1/Conv2D/ReadVariableOp?
+autoencoder_cnn/encoder_layer_extra1/Conv2DConv2D(autoencoder_cnn/reshape/Reshape:output:0Bautoencoder_cnn/encoder_layer_extra1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2-
+autoencoder_cnn/encoder_layer_extra1/Conv2D?
;autoencoder_cnn/encoder_layer_extra1/BiasAdd/ReadVariableOpReadVariableOpDautoencoder_cnn_encoder_layer_extra1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02=
;autoencoder_cnn/encoder_layer_extra1/BiasAdd/ReadVariableOp?
,autoencoder_cnn/encoder_layer_extra1/BiasAddBiasAdd4autoencoder_cnn/encoder_layer_extra1/Conv2D:output:0Cautoencoder_cnn/encoder_layer_extra1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2.
,autoencoder_cnn/encoder_layer_extra1/BiasAdd?
)autoencoder_cnn/encoder_layer_extra1/ReluRelu5autoencoder_cnn/encoder_layer_extra1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2+
)autoencoder_cnn/encoder_layer_extra1/Relu?
:autoencoder_cnn/encoder_layer_extra2/Conv2D/ReadVariableOpReadVariableOpCautoencoder_cnn_encoder_layer_extra2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02<
:autoencoder_cnn/encoder_layer_extra2/Conv2D/ReadVariableOp?
+autoencoder_cnn/encoder_layer_extra2/Conv2DConv2D7autoencoder_cnn/encoder_layer_extra1/Relu:activations:0Bautoencoder_cnn/encoder_layer_extra2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2-
+autoencoder_cnn/encoder_layer_extra2/Conv2D?
;autoencoder_cnn/encoder_layer_extra2/BiasAdd/ReadVariableOpReadVariableOpDautoencoder_cnn_encoder_layer_extra2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02=
;autoencoder_cnn/encoder_layer_extra2/BiasAdd/ReadVariableOp?
,autoencoder_cnn/encoder_layer_extra2/BiasAddBiasAdd4autoencoder_cnn/encoder_layer_extra2/Conv2D:output:0Cautoencoder_cnn/encoder_layer_extra2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2.
,autoencoder_cnn/encoder_layer_extra2/BiasAdd?
)autoencoder_cnn/encoder_layer_extra2/ReluRelu5autoencoder_cnn/encoder_layer_extra2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2+
)autoencoder_cnn/encoder_layer_extra2/Relu?
5autoencoder_cnn/encoder_layer_1/Conv2D/ReadVariableOpReadVariableOp>autoencoder_cnn_encoder_layer_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype027
5autoencoder_cnn/encoder_layer_1/Conv2D/ReadVariableOp?
&autoencoder_cnn/encoder_layer_1/Conv2DConv2D7autoencoder_cnn/encoder_layer_extra2/Relu:activations:0=autoencoder_cnn/encoder_layer_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2(
&autoencoder_cnn/encoder_layer_1/Conv2D?
6autoencoder_cnn/encoder_layer_1/BiasAdd/ReadVariableOpReadVariableOp?autoencoder_cnn_encoder_layer_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype028
6autoencoder_cnn/encoder_layer_1/BiasAdd/ReadVariableOp?
'autoencoder_cnn/encoder_layer_1/BiasAddBiasAdd/autoencoder_cnn/encoder_layer_1/Conv2D:output:0>autoencoder_cnn/encoder_layer_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2)
'autoencoder_cnn/encoder_layer_1/BiasAdd?
$autoencoder_cnn/encoder_layer_1/ReluRelu0autoencoder_cnn/encoder_layer_1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2&
$autoencoder_cnn/encoder_layer_1/Relu?
autoencoder_cnn/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
autoencoder_cnn/flatten/Const?
autoencoder_cnn/flatten/ReshapeReshape2autoencoder_cnn/encoder_layer_1/Relu:activations:0&autoencoder_cnn/flatten/Const:output:0*
T0*(
_output_shapes
:??????????2!
autoencoder_cnn/flatten/Reshape?
*autoencoder_cnn/code/MatMul/ReadVariableOpReadVariableOp3autoencoder_cnn_code_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02,
*autoencoder_cnn/code/MatMul/ReadVariableOp?
autoencoder_cnn/code/MatMulMatMul(autoencoder_cnn/flatten/Reshape:output:02autoencoder_cnn/code/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
autoencoder_cnn/code/MatMul?
+autoencoder_cnn/code/BiasAdd/ReadVariableOpReadVariableOp4autoencoder_cnn_code_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+autoencoder_cnn/code/BiasAdd/ReadVariableOp?
autoencoder_cnn/code/BiasAddBiasAdd%autoencoder_cnn/code/MatMul:product:03autoencoder_cnn/code/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
autoencoder_cnn/code/BiasAdd?
autoencoder_cnn/reshape_1/ShapeShape%autoencoder_cnn/code/BiasAdd:output:0*
T0*
_output_shapes
:2!
autoencoder_cnn/reshape_1/Shape?
-autoencoder_cnn/reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2/
-autoencoder_cnn/reshape_1/strided_slice/stack?
/autoencoder_cnn/reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:21
/autoencoder_cnn/reshape_1/strided_slice/stack_1?
/autoencoder_cnn/reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/autoencoder_cnn/reshape_1/strided_slice/stack_2?
'autoencoder_cnn/reshape_1/strided_sliceStridedSlice(autoencoder_cnn/reshape_1/Shape:output:06autoencoder_cnn/reshape_1/strided_slice/stack:output:08autoencoder_cnn/reshape_1/strided_slice/stack_1:output:08autoencoder_cnn/reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2)
'autoencoder_cnn/reshape_1/strided_slice?
)autoencoder_cnn/reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2+
)autoencoder_cnn/reshape_1/Reshape/shape/1?
)autoencoder_cnn/reshape_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2+
)autoencoder_cnn/reshape_1/Reshape/shape/2?
)autoencoder_cnn/reshape_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2+
)autoencoder_cnn/reshape_1/Reshape/shape/3?
'autoencoder_cnn/reshape_1/Reshape/shapePack0autoencoder_cnn/reshape_1/strided_slice:output:02autoencoder_cnn/reshape_1/Reshape/shape/1:output:02autoencoder_cnn/reshape_1/Reshape/shape/2:output:02autoencoder_cnn/reshape_1/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2)
'autoencoder_cnn/reshape_1/Reshape/shape?
!autoencoder_cnn/reshape_1/ReshapeReshape%autoencoder_cnn/code/BiasAdd:output:00autoencoder_cnn/reshape_1/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????2#
!autoencoder_cnn/reshape_1/Reshape?
*autoencoder_cnn/decoder_layer_extra1/ShapeShape*autoencoder_cnn/reshape_1/Reshape:output:0*
T0*
_output_shapes
:2,
*autoencoder_cnn/decoder_layer_extra1/Shape?
8autoencoder_cnn/decoder_layer_extra1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2:
8autoencoder_cnn/decoder_layer_extra1/strided_slice/stack?
:autoencoder_cnn/decoder_layer_extra1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2<
:autoencoder_cnn/decoder_layer_extra1/strided_slice/stack_1?
:autoencoder_cnn/decoder_layer_extra1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2<
:autoencoder_cnn/decoder_layer_extra1/strided_slice/stack_2?
2autoencoder_cnn/decoder_layer_extra1/strided_sliceStridedSlice3autoencoder_cnn/decoder_layer_extra1/Shape:output:0Aautoencoder_cnn/decoder_layer_extra1/strided_slice/stack:output:0Cautoencoder_cnn/decoder_layer_extra1/strided_slice/stack_1:output:0Cautoencoder_cnn/decoder_layer_extra1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask24
2autoencoder_cnn/decoder_layer_extra1/strided_slice?
,autoencoder_cnn/decoder_layer_extra1/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2.
,autoencoder_cnn/decoder_layer_extra1/stack/1?
,autoencoder_cnn/decoder_layer_extra1/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2.
,autoencoder_cnn/decoder_layer_extra1/stack/2?
,autoencoder_cnn/decoder_layer_extra1/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2.
,autoencoder_cnn/decoder_layer_extra1/stack/3?
*autoencoder_cnn/decoder_layer_extra1/stackPack;autoencoder_cnn/decoder_layer_extra1/strided_slice:output:05autoencoder_cnn/decoder_layer_extra1/stack/1:output:05autoencoder_cnn/decoder_layer_extra1/stack/2:output:05autoencoder_cnn/decoder_layer_extra1/stack/3:output:0*
N*
T0*
_output_shapes
:2,
*autoencoder_cnn/decoder_layer_extra1/stack?
:autoencoder_cnn/decoder_layer_extra1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2<
:autoencoder_cnn/decoder_layer_extra1/strided_slice_1/stack?
<autoencoder_cnn/decoder_layer_extra1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2>
<autoencoder_cnn/decoder_layer_extra1/strided_slice_1/stack_1?
<autoencoder_cnn/decoder_layer_extra1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2>
<autoencoder_cnn/decoder_layer_extra1/strided_slice_1/stack_2?
4autoencoder_cnn/decoder_layer_extra1/strided_slice_1StridedSlice3autoencoder_cnn/decoder_layer_extra1/stack:output:0Cautoencoder_cnn/decoder_layer_extra1/strided_slice_1/stack:output:0Eautoencoder_cnn/decoder_layer_extra1/strided_slice_1/stack_1:output:0Eautoencoder_cnn/decoder_layer_extra1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask26
4autoencoder_cnn/decoder_layer_extra1/strided_slice_1?
Dautoencoder_cnn/decoder_layer_extra1/conv2d_transpose/ReadVariableOpReadVariableOpMautoencoder_cnn_decoder_layer_extra1_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype02F
Dautoencoder_cnn/decoder_layer_extra1/conv2d_transpose/ReadVariableOp?
5autoencoder_cnn/decoder_layer_extra1/conv2d_transposeConv2DBackpropInput3autoencoder_cnn/decoder_layer_extra1/stack:output:0Lautoencoder_cnn/decoder_layer_extra1/conv2d_transpose/ReadVariableOp:value:0*autoencoder_cnn/reshape_1/Reshape:output:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
27
5autoencoder_cnn/decoder_layer_extra1/conv2d_transpose?
;autoencoder_cnn/decoder_layer_extra1/BiasAdd/ReadVariableOpReadVariableOpDautoencoder_cnn_decoder_layer_extra1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02=
;autoencoder_cnn/decoder_layer_extra1/BiasAdd/ReadVariableOp?
,autoencoder_cnn/decoder_layer_extra1/BiasAddBiasAdd>autoencoder_cnn/decoder_layer_extra1/conv2d_transpose:output:0Cautoencoder_cnn/decoder_layer_extra1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2.
,autoencoder_cnn/decoder_layer_extra1/BiasAdd?
)autoencoder_cnn/decoder_layer_extra1/ReluRelu5autoencoder_cnn/decoder_layer_extra1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2+
)autoencoder_cnn/decoder_layer_extra1/Relu?
*autoencoder_cnn/decoder_layer_extra2/ShapeShape7autoencoder_cnn/decoder_layer_extra1/Relu:activations:0*
T0*
_output_shapes
:2,
*autoencoder_cnn/decoder_layer_extra2/Shape?
8autoencoder_cnn/decoder_layer_extra2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2:
8autoencoder_cnn/decoder_layer_extra2/strided_slice/stack?
:autoencoder_cnn/decoder_layer_extra2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2<
:autoencoder_cnn/decoder_layer_extra2/strided_slice/stack_1?
:autoencoder_cnn/decoder_layer_extra2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2<
:autoencoder_cnn/decoder_layer_extra2/strided_slice/stack_2?
2autoencoder_cnn/decoder_layer_extra2/strided_sliceStridedSlice3autoencoder_cnn/decoder_layer_extra2/Shape:output:0Aautoencoder_cnn/decoder_layer_extra2/strided_slice/stack:output:0Cautoencoder_cnn/decoder_layer_extra2/strided_slice/stack_1:output:0Cautoencoder_cnn/decoder_layer_extra2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask24
2autoencoder_cnn/decoder_layer_extra2/strided_slice?
,autoencoder_cnn/decoder_layer_extra2/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2.
,autoencoder_cnn/decoder_layer_extra2/stack/1?
,autoencoder_cnn/decoder_layer_extra2/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2.
,autoencoder_cnn/decoder_layer_extra2/stack/2?
,autoencoder_cnn/decoder_layer_extra2/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2.
,autoencoder_cnn/decoder_layer_extra2/stack/3?
*autoencoder_cnn/decoder_layer_extra2/stackPack;autoencoder_cnn/decoder_layer_extra2/strided_slice:output:05autoencoder_cnn/decoder_layer_extra2/stack/1:output:05autoencoder_cnn/decoder_layer_extra2/stack/2:output:05autoencoder_cnn/decoder_layer_extra2/stack/3:output:0*
N*
T0*
_output_shapes
:2,
*autoencoder_cnn/decoder_layer_extra2/stack?
:autoencoder_cnn/decoder_layer_extra2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2<
:autoencoder_cnn/decoder_layer_extra2/strided_slice_1/stack?
<autoencoder_cnn/decoder_layer_extra2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2>
<autoencoder_cnn/decoder_layer_extra2/strided_slice_1/stack_1?
<autoencoder_cnn/decoder_layer_extra2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2>
<autoencoder_cnn/decoder_layer_extra2/strided_slice_1/stack_2?
4autoencoder_cnn/decoder_layer_extra2/strided_slice_1StridedSlice3autoencoder_cnn/decoder_layer_extra2/stack:output:0Cautoencoder_cnn/decoder_layer_extra2/strided_slice_1/stack:output:0Eautoencoder_cnn/decoder_layer_extra2/strided_slice_1/stack_1:output:0Eautoencoder_cnn/decoder_layer_extra2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask26
4autoencoder_cnn/decoder_layer_extra2/strided_slice_1?
Dautoencoder_cnn/decoder_layer_extra2/conv2d_transpose/ReadVariableOpReadVariableOpMautoencoder_cnn_decoder_layer_extra2_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype02F
Dautoencoder_cnn/decoder_layer_extra2/conv2d_transpose/ReadVariableOp?
5autoencoder_cnn/decoder_layer_extra2/conv2d_transposeConv2DBackpropInput3autoencoder_cnn/decoder_layer_extra2/stack:output:0Lautoencoder_cnn/decoder_layer_extra2/conv2d_transpose/ReadVariableOp:value:07autoencoder_cnn/decoder_layer_extra1/Relu:activations:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
27
5autoencoder_cnn/decoder_layer_extra2/conv2d_transpose?
;autoencoder_cnn/decoder_layer_extra2/BiasAdd/ReadVariableOpReadVariableOpDautoencoder_cnn_decoder_layer_extra2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02=
;autoencoder_cnn/decoder_layer_extra2/BiasAdd/ReadVariableOp?
,autoencoder_cnn/decoder_layer_extra2/BiasAddBiasAdd>autoencoder_cnn/decoder_layer_extra2/conv2d_transpose:output:0Cautoencoder_cnn/decoder_layer_extra2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2.
,autoencoder_cnn/decoder_layer_extra2/BiasAdd?
)autoencoder_cnn/decoder_layer_extra2/ReluRelu5autoencoder_cnn/decoder_layer_extra2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2+
)autoencoder_cnn/decoder_layer_extra2/Relu?
%autoencoder_cnn/decoder_layer_1/ShapeShape7autoencoder_cnn/decoder_layer_extra2/Relu:activations:0*
T0*
_output_shapes
:2'
%autoencoder_cnn/decoder_layer_1/Shape?
3autoencoder_cnn/decoder_layer_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 25
3autoencoder_cnn/decoder_layer_1/strided_slice/stack?
5autoencoder_cnn/decoder_layer_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5autoencoder_cnn/decoder_layer_1/strided_slice/stack_1?
5autoencoder_cnn/decoder_layer_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5autoencoder_cnn/decoder_layer_1/strided_slice/stack_2?
-autoencoder_cnn/decoder_layer_1/strided_sliceStridedSlice.autoencoder_cnn/decoder_layer_1/Shape:output:0<autoencoder_cnn/decoder_layer_1/strided_slice/stack:output:0>autoencoder_cnn/decoder_layer_1/strided_slice/stack_1:output:0>autoencoder_cnn/decoder_layer_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-autoencoder_cnn/decoder_layer_1/strided_slice?
'autoencoder_cnn/decoder_layer_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2)
'autoencoder_cnn/decoder_layer_1/stack/1?
'autoencoder_cnn/decoder_layer_1/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2)
'autoencoder_cnn/decoder_layer_1/stack/2?
'autoencoder_cnn/decoder_layer_1/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2)
'autoencoder_cnn/decoder_layer_1/stack/3?
%autoencoder_cnn/decoder_layer_1/stackPack6autoencoder_cnn/decoder_layer_1/strided_slice:output:00autoencoder_cnn/decoder_layer_1/stack/1:output:00autoencoder_cnn/decoder_layer_1/stack/2:output:00autoencoder_cnn/decoder_layer_1/stack/3:output:0*
N*
T0*
_output_shapes
:2'
%autoencoder_cnn/decoder_layer_1/stack?
5autoencoder_cnn/decoder_layer_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 27
5autoencoder_cnn/decoder_layer_1/strided_slice_1/stack?
7autoencoder_cnn/decoder_layer_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7autoencoder_cnn/decoder_layer_1/strided_slice_1/stack_1?
7autoencoder_cnn/decoder_layer_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7autoencoder_cnn/decoder_layer_1/strided_slice_1/stack_2?
/autoencoder_cnn/decoder_layer_1/strided_slice_1StridedSlice.autoencoder_cnn/decoder_layer_1/stack:output:0>autoencoder_cnn/decoder_layer_1/strided_slice_1/stack:output:0@autoencoder_cnn/decoder_layer_1/strided_slice_1/stack_1:output:0@autoencoder_cnn/decoder_layer_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/autoencoder_cnn/decoder_layer_1/strided_slice_1?
?autoencoder_cnn/decoder_layer_1/conv2d_transpose/ReadVariableOpReadVariableOpHautoencoder_cnn_decoder_layer_1_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype02A
?autoencoder_cnn/decoder_layer_1/conv2d_transpose/ReadVariableOp?
0autoencoder_cnn/decoder_layer_1/conv2d_transposeConv2DBackpropInput.autoencoder_cnn/decoder_layer_1/stack:output:0Gautoencoder_cnn/decoder_layer_1/conv2d_transpose/ReadVariableOp:value:07autoencoder_cnn/decoder_layer_extra2/Relu:activations:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
22
0autoencoder_cnn/decoder_layer_1/conv2d_transpose?
6autoencoder_cnn/decoder_layer_1/BiasAdd/ReadVariableOpReadVariableOp?autoencoder_cnn_decoder_layer_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype028
6autoencoder_cnn/decoder_layer_1/BiasAdd/ReadVariableOp?
'autoencoder_cnn/decoder_layer_1/BiasAddBiasAdd9autoencoder_cnn/decoder_layer_1/conv2d_transpose:output:0>autoencoder_cnn/decoder_layer_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2)
'autoencoder_cnn/decoder_layer_1/BiasAdd?
$autoencoder_cnn/decoder_layer_1/ReluRelu0autoencoder_cnn/decoder_layer_1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2&
$autoencoder_cnn/decoder_layer_1/Relu?
%autoencoder_cnn/decoder_layer_2/ShapeShape2autoencoder_cnn/decoder_layer_1/Relu:activations:0*
T0*
_output_shapes
:2'
%autoencoder_cnn/decoder_layer_2/Shape?
3autoencoder_cnn/decoder_layer_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 25
3autoencoder_cnn/decoder_layer_2/strided_slice/stack?
5autoencoder_cnn/decoder_layer_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5autoencoder_cnn/decoder_layer_2/strided_slice/stack_1?
5autoencoder_cnn/decoder_layer_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5autoencoder_cnn/decoder_layer_2/strided_slice/stack_2?
-autoencoder_cnn/decoder_layer_2/strided_sliceStridedSlice.autoencoder_cnn/decoder_layer_2/Shape:output:0<autoencoder_cnn/decoder_layer_2/strided_slice/stack:output:0>autoencoder_cnn/decoder_layer_2/strided_slice/stack_1:output:0>autoencoder_cnn/decoder_layer_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-autoencoder_cnn/decoder_layer_2/strided_slice?
'autoencoder_cnn/decoder_layer_2/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2)
'autoencoder_cnn/decoder_layer_2/stack/1?
'autoencoder_cnn/decoder_layer_2/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2)
'autoencoder_cnn/decoder_layer_2/stack/2?
'autoencoder_cnn/decoder_layer_2/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2)
'autoencoder_cnn/decoder_layer_2/stack/3?
%autoencoder_cnn/decoder_layer_2/stackPack6autoencoder_cnn/decoder_layer_2/strided_slice:output:00autoencoder_cnn/decoder_layer_2/stack/1:output:00autoencoder_cnn/decoder_layer_2/stack/2:output:00autoencoder_cnn/decoder_layer_2/stack/3:output:0*
N*
T0*
_output_shapes
:2'
%autoencoder_cnn/decoder_layer_2/stack?
5autoencoder_cnn/decoder_layer_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 27
5autoencoder_cnn/decoder_layer_2/strided_slice_1/stack?
7autoencoder_cnn/decoder_layer_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7autoencoder_cnn/decoder_layer_2/strided_slice_1/stack_1?
7autoencoder_cnn/decoder_layer_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7autoencoder_cnn/decoder_layer_2/strided_slice_1/stack_2?
/autoencoder_cnn/decoder_layer_2/strided_slice_1StridedSlice.autoencoder_cnn/decoder_layer_2/stack:output:0>autoencoder_cnn/decoder_layer_2/strided_slice_1/stack:output:0@autoencoder_cnn/decoder_layer_2/strided_slice_1/stack_1:output:0@autoencoder_cnn/decoder_layer_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/autoencoder_cnn/decoder_layer_2/strided_slice_1?
?autoencoder_cnn/decoder_layer_2/conv2d_transpose/ReadVariableOpReadVariableOpHautoencoder_cnn_decoder_layer_2_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype02A
?autoencoder_cnn/decoder_layer_2/conv2d_transpose/ReadVariableOp?
0autoencoder_cnn/decoder_layer_2/conv2d_transposeConv2DBackpropInput.autoencoder_cnn/decoder_layer_2/stack:output:0Gautoencoder_cnn/decoder_layer_2/conv2d_transpose/ReadVariableOp:value:02autoencoder_cnn/decoder_layer_1/Relu:activations:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
22
0autoencoder_cnn/decoder_layer_2/conv2d_transpose?
6autoencoder_cnn/decoder_layer_2/BiasAdd/ReadVariableOpReadVariableOp?autoencoder_cnn_decoder_layer_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype028
6autoencoder_cnn/decoder_layer_2/BiasAdd/ReadVariableOp?
'autoencoder_cnn/decoder_layer_2/BiasAddBiasAdd9autoencoder_cnn/decoder_layer_2/conv2d_transpose:output:0>autoencoder_cnn/decoder_layer_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2)
'autoencoder_cnn/decoder_layer_2/BiasAdd?
$autoencoder_cnn/decoder_layer_2/ReluRelu0autoencoder_cnn/decoder_layer_2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2&
$autoencoder_cnn/decoder_layer_2/Relu?
IdentityIdentity2autoencoder_cnn/decoder_layer_2/Relu:activations:0,^autoencoder_cnn/code/BiasAdd/ReadVariableOp+^autoencoder_cnn/code/MatMul/ReadVariableOp7^autoencoder_cnn/decoder_layer_1/BiasAdd/ReadVariableOp@^autoencoder_cnn/decoder_layer_1/conv2d_transpose/ReadVariableOp7^autoencoder_cnn/decoder_layer_2/BiasAdd/ReadVariableOp@^autoencoder_cnn/decoder_layer_2/conv2d_transpose/ReadVariableOp<^autoencoder_cnn/decoder_layer_extra1/BiasAdd/ReadVariableOpE^autoencoder_cnn/decoder_layer_extra1/conv2d_transpose/ReadVariableOp<^autoencoder_cnn/decoder_layer_extra2/BiasAdd/ReadVariableOpE^autoencoder_cnn/decoder_layer_extra2/conv2d_transpose/ReadVariableOp7^autoencoder_cnn/encoder_layer_1/BiasAdd/ReadVariableOp6^autoencoder_cnn/encoder_layer_1/Conv2D/ReadVariableOp<^autoencoder_cnn/encoder_layer_extra1/BiasAdd/ReadVariableOp;^autoencoder_cnn/encoder_layer_extra1/Conv2D/ReadVariableOp<^autoencoder_cnn/encoder_layer_extra2/BiasAdd/ReadVariableOp;^autoencoder_cnn/encoder_layer_extra2/Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:?????????: : : : : : : : : : : : : : : : 2Z
+autoencoder_cnn/code/BiasAdd/ReadVariableOp+autoencoder_cnn/code/BiasAdd/ReadVariableOp2X
*autoencoder_cnn/code/MatMul/ReadVariableOp*autoencoder_cnn/code/MatMul/ReadVariableOp2p
6autoencoder_cnn/decoder_layer_1/BiasAdd/ReadVariableOp6autoencoder_cnn/decoder_layer_1/BiasAdd/ReadVariableOp2?
?autoencoder_cnn/decoder_layer_1/conv2d_transpose/ReadVariableOp?autoencoder_cnn/decoder_layer_1/conv2d_transpose/ReadVariableOp2p
6autoencoder_cnn/decoder_layer_2/BiasAdd/ReadVariableOp6autoencoder_cnn/decoder_layer_2/BiasAdd/ReadVariableOp2?
?autoencoder_cnn/decoder_layer_2/conv2d_transpose/ReadVariableOp?autoencoder_cnn/decoder_layer_2/conv2d_transpose/ReadVariableOp2z
;autoencoder_cnn/decoder_layer_extra1/BiasAdd/ReadVariableOp;autoencoder_cnn/decoder_layer_extra1/BiasAdd/ReadVariableOp2?
Dautoencoder_cnn/decoder_layer_extra1/conv2d_transpose/ReadVariableOpDautoencoder_cnn/decoder_layer_extra1/conv2d_transpose/ReadVariableOp2z
;autoencoder_cnn/decoder_layer_extra2/BiasAdd/ReadVariableOp;autoencoder_cnn/decoder_layer_extra2/BiasAdd/ReadVariableOp2?
Dautoencoder_cnn/decoder_layer_extra2/conv2d_transpose/ReadVariableOpDautoencoder_cnn/decoder_layer_extra2/conv2d_transpose/ReadVariableOp2p
6autoencoder_cnn/encoder_layer_1/BiasAdd/ReadVariableOp6autoencoder_cnn/encoder_layer_1/BiasAdd/ReadVariableOp2n
5autoencoder_cnn/encoder_layer_1/Conv2D/ReadVariableOp5autoencoder_cnn/encoder_layer_1/Conv2D/ReadVariableOp2z
;autoencoder_cnn/encoder_layer_extra1/BiasAdd/ReadVariableOp;autoencoder_cnn/encoder_layer_extra1/BiasAdd/ReadVariableOp2x
:autoencoder_cnn/encoder_layer_extra1/Conv2D/ReadVariableOp:autoencoder_cnn/encoder_layer_extra1/Conv2D/ReadVariableOp2z
;autoencoder_cnn/encoder_layer_extra2/BiasAdd/ReadVariableOp;autoencoder_cnn/encoder_layer_extra2/BiasAdd/ReadVariableOp2x
:autoencoder_cnn/encoder_layer_extra2/Conv2D/ReadVariableOp:autoencoder_cnn/encoder_layer_extra2/Conv2D/ReadVariableOp:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_1
?%
?
O__inference_decoder_layer_extra1_layer_call_and_return_conditional_losses_12479

inputsB
(conv2d_transpose_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1T
stack/3Const*
_output_shapes
: *
dtype0*
value	B :2	
stack/3?
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_transpose/ReadVariableOp?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingSAME*
strides
2
conv2d_transpose?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
^
B__inference_reshape_layer_call_and_return_conditional_losses_13503

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/2d
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/3?
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapew
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:?????????2	
Reshapel
IdentityIdentityReshape:output:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
C
'__inference_flatten_layer_call_fn_13579

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_127042
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
J__inference_encoder_layer_1_layer_call_and_return_conditional_losses_12692

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
#__inference_signature_wrapper_13149
input_1!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:#
	unknown_3:
	unknown_4:
	unknown_5:	?
	unknown_6:#
	unknown_7:
	unknown_8:#
	unknown_9:

unknown_10:$

unknown_11:

unknown_12:$

unknown_13:

unknown_14:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *)
f$R"
 __inference__wrapped_model_124442
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:?????????: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_1
?
?
O__inference_encoder_layer_extra1_layer_call_and_return_conditional_losses_13519

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?%
?
J__inference_decoder_layer_1_layer_call_and_return_conditional_losses_12569

inputsB
(conv2d_transpose_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1T
stack/3Const*
_output_shapes
: *
dtype0*
value	B :2	
stack/3?
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_transpose/ReadVariableOp?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingSAME*
strides
2
conv2d_transpose?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
4__inference_encoder_layer_extra2_layer_call_fn_13548

inputs!
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_encoder_layer_extra2_layer_call_and_return_conditional_losses_126752
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?%
?
O__inference_decoder_layer_extra2_layer_call_and_return_conditional_losses_12524

inputsB
(conv2d_transpose_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1T
stack/3Const*
_output_shapes
: *
dtype0*
value	B :2	
stack/3?
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_transpose/ReadVariableOp?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingSAME*
strides
2
conv2d_transpose?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
/__inference_encoder_layer_1_layer_call_fn_13568

inputs!
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_encoder_layer_1_layer_call_and_return_conditional_losses_126922
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?<
?
J__inference_autoencoder_cnn_layer_call_and_return_conditional_losses_12759

inputs4
encoder_layer_extra1_12659:(
encoder_layer_extra1_12661:4
encoder_layer_extra2_12676:(
encoder_layer_extra2_12678:/
encoder_layer_1_12693:#
encoder_layer_1_12695:

code_12717:	?

code_12719:4
decoder_layer_extra1_12738:(
decoder_layer_extra1_12740:4
decoder_layer_extra2_12743:(
decoder_layer_extra2_12745:/
decoder_layer_1_12748:#
decoder_layer_1_12750:/
decoder_layer_2_12753:#
decoder_layer_2_12755:
identity??code/StatefulPartitionedCall?'decoder_layer_1/StatefulPartitionedCall?'decoder_layer_2/StatefulPartitionedCall?,decoder_layer_extra1/StatefulPartitionedCall?,decoder_layer_extra2/StatefulPartitionedCall?'encoder_layer_1/StatefulPartitionedCall?,encoder_layer_extra1/StatefulPartitionedCall?,encoder_layer_extra2/StatefulPartitionedCall?
reshape/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_reshape_layer_call_and_return_conditional_losses_126452
reshape/PartitionedCall?
,encoder_layer_extra1/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0encoder_layer_extra1_12659encoder_layer_extra1_12661*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_encoder_layer_extra1_layer_call_and_return_conditional_losses_126582.
,encoder_layer_extra1/StatefulPartitionedCall?
,encoder_layer_extra2/StatefulPartitionedCallStatefulPartitionedCall5encoder_layer_extra1/StatefulPartitionedCall:output:0encoder_layer_extra2_12676encoder_layer_extra2_12678*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_encoder_layer_extra2_layer_call_and_return_conditional_losses_126752.
,encoder_layer_extra2/StatefulPartitionedCall?
'encoder_layer_1/StatefulPartitionedCallStatefulPartitionedCall5encoder_layer_extra2/StatefulPartitionedCall:output:0encoder_layer_1_12693encoder_layer_1_12695*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_encoder_layer_1_layer_call_and_return_conditional_losses_126922)
'encoder_layer_1/StatefulPartitionedCall?
flatten/PartitionedCallPartitionedCall0encoder_layer_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_127042
flatten/PartitionedCall?
code/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0
code_12717
code_12719*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_code_layer_call_and_return_conditional_losses_127162
code/StatefulPartitionedCall?
reshape_1/PartitionedCallPartitionedCall%code/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_reshape_1_layer_call_and_return_conditional_losses_127362
reshape_1/PartitionedCall?
,decoder_layer_extra1/StatefulPartitionedCallStatefulPartitionedCall"reshape_1/PartitionedCall:output:0decoder_layer_extra1_12738decoder_layer_extra1_12740*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_decoder_layer_extra1_layer_call_and_return_conditional_losses_124792.
,decoder_layer_extra1/StatefulPartitionedCall?
,decoder_layer_extra2/StatefulPartitionedCallStatefulPartitionedCall5decoder_layer_extra1/StatefulPartitionedCall:output:0decoder_layer_extra2_12743decoder_layer_extra2_12745*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_decoder_layer_extra2_layer_call_and_return_conditional_losses_125242.
,decoder_layer_extra2/StatefulPartitionedCall?
'decoder_layer_1/StatefulPartitionedCallStatefulPartitionedCall5decoder_layer_extra2/StatefulPartitionedCall:output:0decoder_layer_1_12748decoder_layer_1_12750*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_decoder_layer_1_layer_call_and_return_conditional_losses_125692)
'decoder_layer_1/StatefulPartitionedCall?
'decoder_layer_2/StatefulPartitionedCallStatefulPartitionedCall0decoder_layer_1/StatefulPartitionedCall:output:0decoder_layer_2_12753decoder_layer_2_12755*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_decoder_layer_2_layer_call_and_return_conditional_losses_126142)
'decoder_layer_2/StatefulPartitionedCall?
IdentityIdentity0decoder_layer_2/StatefulPartitionedCall:output:0^code/StatefulPartitionedCall(^decoder_layer_1/StatefulPartitionedCall(^decoder_layer_2/StatefulPartitionedCall-^decoder_layer_extra1/StatefulPartitionedCall-^decoder_layer_extra2/StatefulPartitionedCall(^encoder_layer_1/StatefulPartitionedCall-^encoder_layer_extra1/StatefulPartitionedCall-^encoder_layer_extra2/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:?????????: : : : : : : : : : : : : : : : 2<
code/StatefulPartitionedCallcode/StatefulPartitionedCall2R
'decoder_layer_1/StatefulPartitionedCall'decoder_layer_1/StatefulPartitionedCall2R
'decoder_layer_2/StatefulPartitionedCall'decoder_layer_2/StatefulPartitionedCall2\
,decoder_layer_extra1/StatefulPartitionedCall,decoder_layer_extra1/StatefulPartitionedCall2\
,decoder_layer_extra2/StatefulPartitionedCall,decoder_layer_extra2/StatefulPartitionedCall2R
'encoder_layer_1/StatefulPartitionedCall'encoder_layer_1/StatefulPartitionedCall2\
,encoder_layer_extra1/StatefulPartitionedCall,encoder_layer_extra1/StatefulPartitionedCall2\
,encoder_layer_extra2/StatefulPartitionedCall,encoder_layer_extra2/StatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
C
'__inference_reshape_layer_call_fn_13508

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_reshape_layer_call_and_return_conditional_losses_126452
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
O__inference_encoder_layer_extra2_layer_call_and_return_conditional_losses_12675

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
?__inference_code_layer_call_and_return_conditional_losses_13589

inputs1
matmul_readvariableop_resource:	?-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
^
B__inference_flatten_layer_call_and_return_conditional_losses_13574

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?
J__inference_autoencoder_cnn_layer_call_and_return_conditional_losses_13415

inputsM
3encoder_layer_extra1_conv2d_readvariableop_resource:B
4encoder_layer_extra1_biasadd_readvariableop_resource:M
3encoder_layer_extra2_conv2d_readvariableop_resource:B
4encoder_layer_extra2_biasadd_readvariableop_resource:H
.encoder_layer_1_conv2d_readvariableop_resource:=
/encoder_layer_1_biasadd_readvariableop_resource:6
#code_matmul_readvariableop_resource:	?2
$code_biasadd_readvariableop_resource:W
=decoder_layer_extra1_conv2d_transpose_readvariableop_resource:B
4decoder_layer_extra1_biasadd_readvariableop_resource:W
=decoder_layer_extra2_conv2d_transpose_readvariableop_resource:B
4decoder_layer_extra2_biasadd_readvariableop_resource:R
8decoder_layer_1_conv2d_transpose_readvariableop_resource:=
/decoder_layer_1_biasadd_readvariableop_resource:R
8decoder_layer_2_conv2d_transpose_readvariableop_resource:=
/decoder_layer_2_biasadd_readvariableop_resource:
identity??code/BiasAdd/ReadVariableOp?code/MatMul/ReadVariableOp?&decoder_layer_1/BiasAdd/ReadVariableOp?/decoder_layer_1/conv2d_transpose/ReadVariableOp?&decoder_layer_2/BiasAdd/ReadVariableOp?/decoder_layer_2/conv2d_transpose/ReadVariableOp?+decoder_layer_extra1/BiasAdd/ReadVariableOp?4decoder_layer_extra1/conv2d_transpose/ReadVariableOp?+decoder_layer_extra2/BiasAdd/ReadVariableOp?4decoder_layer_extra2/conv2d_transpose/ReadVariableOp?&encoder_layer_1/BiasAdd/ReadVariableOp?%encoder_layer_1/Conv2D/ReadVariableOp?+encoder_layer_extra1/BiasAdd/ReadVariableOp?*encoder_layer_extra1/Conv2D/ReadVariableOp?+encoder_layer_extra2/BiasAdd/ReadVariableOp?*encoder_layer_extra2/Conv2D/ReadVariableOpT
reshape/ShapeShapeinputs*
T0*
_output_shapes
:2
reshape/Shape?
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape/strided_slice/stack?
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_1?
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_2?
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape/strided_slicet
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/1t
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/2t
reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/3?
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0 reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape/Reshape/shape?
reshape/ReshapeReshapeinputsreshape/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????2
reshape/Reshape?
*encoder_layer_extra1/Conv2D/ReadVariableOpReadVariableOp3encoder_layer_extra1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02,
*encoder_layer_extra1/Conv2D/ReadVariableOp?
encoder_layer_extra1/Conv2DConv2Dreshape/Reshape:output:02encoder_layer_extra1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
encoder_layer_extra1/Conv2D?
+encoder_layer_extra1/BiasAdd/ReadVariableOpReadVariableOp4encoder_layer_extra1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+encoder_layer_extra1/BiasAdd/ReadVariableOp?
encoder_layer_extra1/BiasAddBiasAdd$encoder_layer_extra1/Conv2D:output:03encoder_layer_extra1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
encoder_layer_extra1/BiasAdd?
encoder_layer_extra1/ReluRelu%encoder_layer_extra1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
encoder_layer_extra1/Relu?
*encoder_layer_extra2/Conv2D/ReadVariableOpReadVariableOp3encoder_layer_extra2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02,
*encoder_layer_extra2/Conv2D/ReadVariableOp?
encoder_layer_extra2/Conv2DConv2D'encoder_layer_extra1/Relu:activations:02encoder_layer_extra2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
encoder_layer_extra2/Conv2D?
+encoder_layer_extra2/BiasAdd/ReadVariableOpReadVariableOp4encoder_layer_extra2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+encoder_layer_extra2/BiasAdd/ReadVariableOp?
encoder_layer_extra2/BiasAddBiasAdd$encoder_layer_extra2/Conv2D:output:03encoder_layer_extra2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
encoder_layer_extra2/BiasAdd?
encoder_layer_extra2/ReluRelu%encoder_layer_extra2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
encoder_layer_extra2/Relu?
%encoder_layer_1/Conv2D/ReadVariableOpReadVariableOp.encoder_layer_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02'
%encoder_layer_1/Conv2D/ReadVariableOp?
encoder_layer_1/Conv2DConv2D'encoder_layer_extra2/Relu:activations:0-encoder_layer_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
encoder_layer_1/Conv2D?
&encoder_layer_1/BiasAdd/ReadVariableOpReadVariableOp/encoder_layer_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&encoder_layer_1/BiasAdd/ReadVariableOp?
encoder_layer_1/BiasAddBiasAddencoder_layer_1/Conv2D:output:0.encoder_layer_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
encoder_layer_1/BiasAdd?
encoder_layer_1/ReluRelu encoder_layer_1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
encoder_layer_1/Reluo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten/Const?
flatten/ReshapeReshape"encoder_layer_1/Relu:activations:0flatten/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten/Reshape?
code/MatMul/ReadVariableOpReadVariableOp#code_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
code/MatMul/ReadVariableOp?
code/MatMulMatMulflatten/Reshape:output:0"code/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
code/MatMul?
code/BiasAdd/ReadVariableOpReadVariableOp$code_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
code/BiasAdd/ReadVariableOp?
code/BiasAddBiasAddcode/MatMul:product:0#code/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
code/BiasAddg
reshape_1/ShapeShapecode/BiasAdd:output:0*
T0*
_output_shapes
:2
reshape_1/Shape?
reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape_1/strided_slice/stack?
reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_1/strided_slice/stack_1?
reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_1/strided_slice/stack_2?
reshape_1/strided_sliceStridedSlicereshape_1/Shape:output:0&reshape_1/strided_slice/stack:output:0(reshape_1/strided_slice/stack_1:output:0(reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_1/strided_slicex
reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_1/Reshape/shape/1x
reshape_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_1/Reshape/shape/2x
reshape_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_1/Reshape/shape/3?
reshape_1/Reshape/shapePack reshape_1/strided_slice:output:0"reshape_1/Reshape/shape/1:output:0"reshape_1/Reshape/shape/2:output:0"reshape_1/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape_1/Reshape/shape?
reshape_1/ReshapeReshapecode/BiasAdd:output:0 reshape_1/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????2
reshape_1/Reshape?
decoder_layer_extra1/ShapeShapereshape_1/Reshape:output:0*
T0*
_output_shapes
:2
decoder_layer_extra1/Shape?
(decoder_layer_extra1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(decoder_layer_extra1/strided_slice/stack?
*decoder_layer_extra1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*decoder_layer_extra1/strided_slice/stack_1?
*decoder_layer_extra1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*decoder_layer_extra1/strided_slice/stack_2?
"decoder_layer_extra1/strided_sliceStridedSlice#decoder_layer_extra1/Shape:output:01decoder_layer_extra1/strided_slice/stack:output:03decoder_layer_extra1/strided_slice/stack_1:output:03decoder_layer_extra1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"decoder_layer_extra1/strided_slice~
decoder_layer_extra1/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2
decoder_layer_extra1/stack/1~
decoder_layer_extra1/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
decoder_layer_extra1/stack/2~
decoder_layer_extra1/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
decoder_layer_extra1/stack/3?
decoder_layer_extra1/stackPack+decoder_layer_extra1/strided_slice:output:0%decoder_layer_extra1/stack/1:output:0%decoder_layer_extra1/stack/2:output:0%decoder_layer_extra1/stack/3:output:0*
N*
T0*
_output_shapes
:2
decoder_layer_extra1/stack?
*decoder_layer_extra1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*decoder_layer_extra1/strided_slice_1/stack?
,decoder_layer_extra1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,decoder_layer_extra1/strided_slice_1/stack_1?
,decoder_layer_extra1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,decoder_layer_extra1/strided_slice_1/stack_2?
$decoder_layer_extra1/strided_slice_1StridedSlice#decoder_layer_extra1/stack:output:03decoder_layer_extra1/strided_slice_1/stack:output:05decoder_layer_extra1/strided_slice_1/stack_1:output:05decoder_layer_extra1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2&
$decoder_layer_extra1/strided_slice_1?
4decoder_layer_extra1/conv2d_transpose/ReadVariableOpReadVariableOp=decoder_layer_extra1_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype026
4decoder_layer_extra1/conv2d_transpose/ReadVariableOp?
%decoder_layer_extra1/conv2d_transposeConv2DBackpropInput#decoder_layer_extra1/stack:output:0<decoder_layer_extra1/conv2d_transpose/ReadVariableOp:value:0reshape_1/Reshape:output:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2'
%decoder_layer_extra1/conv2d_transpose?
+decoder_layer_extra1/BiasAdd/ReadVariableOpReadVariableOp4decoder_layer_extra1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+decoder_layer_extra1/BiasAdd/ReadVariableOp?
decoder_layer_extra1/BiasAddBiasAdd.decoder_layer_extra1/conv2d_transpose:output:03decoder_layer_extra1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
decoder_layer_extra1/BiasAdd?
decoder_layer_extra1/ReluRelu%decoder_layer_extra1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
decoder_layer_extra1/Relu?
decoder_layer_extra2/ShapeShape'decoder_layer_extra1/Relu:activations:0*
T0*
_output_shapes
:2
decoder_layer_extra2/Shape?
(decoder_layer_extra2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(decoder_layer_extra2/strided_slice/stack?
*decoder_layer_extra2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*decoder_layer_extra2/strided_slice/stack_1?
*decoder_layer_extra2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*decoder_layer_extra2/strided_slice/stack_2?
"decoder_layer_extra2/strided_sliceStridedSlice#decoder_layer_extra2/Shape:output:01decoder_layer_extra2/strided_slice/stack:output:03decoder_layer_extra2/strided_slice/stack_1:output:03decoder_layer_extra2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"decoder_layer_extra2/strided_slice~
decoder_layer_extra2/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2
decoder_layer_extra2/stack/1~
decoder_layer_extra2/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
decoder_layer_extra2/stack/2~
decoder_layer_extra2/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
decoder_layer_extra2/stack/3?
decoder_layer_extra2/stackPack+decoder_layer_extra2/strided_slice:output:0%decoder_layer_extra2/stack/1:output:0%decoder_layer_extra2/stack/2:output:0%decoder_layer_extra2/stack/3:output:0*
N*
T0*
_output_shapes
:2
decoder_layer_extra2/stack?
*decoder_layer_extra2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*decoder_layer_extra2/strided_slice_1/stack?
,decoder_layer_extra2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,decoder_layer_extra2/strided_slice_1/stack_1?
,decoder_layer_extra2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,decoder_layer_extra2/strided_slice_1/stack_2?
$decoder_layer_extra2/strided_slice_1StridedSlice#decoder_layer_extra2/stack:output:03decoder_layer_extra2/strided_slice_1/stack:output:05decoder_layer_extra2/strided_slice_1/stack_1:output:05decoder_layer_extra2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2&
$decoder_layer_extra2/strided_slice_1?
4decoder_layer_extra2/conv2d_transpose/ReadVariableOpReadVariableOp=decoder_layer_extra2_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype026
4decoder_layer_extra2/conv2d_transpose/ReadVariableOp?
%decoder_layer_extra2/conv2d_transposeConv2DBackpropInput#decoder_layer_extra2/stack:output:0<decoder_layer_extra2/conv2d_transpose/ReadVariableOp:value:0'decoder_layer_extra1/Relu:activations:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2'
%decoder_layer_extra2/conv2d_transpose?
+decoder_layer_extra2/BiasAdd/ReadVariableOpReadVariableOp4decoder_layer_extra2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+decoder_layer_extra2/BiasAdd/ReadVariableOp?
decoder_layer_extra2/BiasAddBiasAdd.decoder_layer_extra2/conv2d_transpose:output:03decoder_layer_extra2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
decoder_layer_extra2/BiasAdd?
decoder_layer_extra2/ReluRelu%decoder_layer_extra2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
decoder_layer_extra2/Relu?
decoder_layer_1/ShapeShape'decoder_layer_extra2/Relu:activations:0*
T0*
_output_shapes
:2
decoder_layer_1/Shape?
#decoder_layer_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2%
#decoder_layer_1/strided_slice/stack?
%decoder_layer_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%decoder_layer_1/strided_slice/stack_1?
%decoder_layer_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%decoder_layer_1/strided_slice/stack_2?
decoder_layer_1/strided_sliceStridedSlicedecoder_layer_1/Shape:output:0,decoder_layer_1/strided_slice/stack:output:0.decoder_layer_1/strided_slice/stack_1:output:0.decoder_layer_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
decoder_layer_1/strided_slicet
decoder_layer_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2
decoder_layer_1/stack/1t
decoder_layer_1/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
decoder_layer_1/stack/2t
decoder_layer_1/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
decoder_layer_1/stack/3?
decoder_layer_1/stackPack&decoder_layer_1/strided_slice:output:0 decoder_layer_1/stack/1:output:0 decoder_layer_1/stack/2:output:0 decoder_layer_1/stack/3:output:0*
N*
T0*
_output_shapes
:2
decoder_layer_1/stack?
%decoder_layer_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%decoder_layer_1/strided_slice_1/stack?
'decoder_layer_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'decoder_layer_1/strided_slice_1/stack_1?
'decoder_layer_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'decoder_layer_1/strided_slice_1/stack_2?
decoder_layer_1/strided_slice_1StridedSlicedecoder_layer_1/stack:output:0.decoder_layer_1/strided_slice_1/stack:output:00decoder_layer_1/strided_slice_1/stack_1:output:00decoder_layer_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
decoder_layer_1/strided_slice_1?
/decoder_layer_1/conv2d_transpose/ReadVariableOpReadVariableOp8decoder_layer_1_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype021
/decoder_layer_1/conv2d_transpose/ReadVariableOp?
 decoder_layer_1/conv2d_transposeConv2DBackpropInputdecoder_layer_1/stack:output:07decoder_layer_1/conv2d_transpose/ReadVariableOp:value:0'decoder_layer_extra2/Relu:activations:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2"
 decoder_layer_1/conv2d_transpose?
&decoder_layer_1/BiasAdd/ReadVariableOpReadVariableOp/decoder_layer_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&decoder_layer_1/BiasAdd/ReadVariableOp?
decoder_layer_1/BiasAddBiasAdd)decoder_layer_1/conv2d_transpose:output:0.decoder_layer_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
decoder_layer_1/BiasAdd?
decoder_layer_1/ReluRelu decoder_layer_1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
decoder_layer_1/Relu?
decoder_layer_2/ShapeShape"decoder_layer_1/Relu:activations:0*
T0*
_output_shapes
:2
decoder_layer_2/Shape?
#decoder_layer_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2%
#decoder_layer_2/strided_slice/stack?
%decoder_layer_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%decoder_layer_2/strided_slice/stack_1?
%decoder_layer_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%decoder_layer_2/strided_slice/stack_2?
decoder_layer_2/strided_sliceStridedSlicedecoder_layer_2/Shape:output:0,decoder_layer_2/strided_slice/stack:output:0.decoder_layer_2/strided_slice/stack_1:output:0.decoder_layer_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
decoder_layer_2/strided_slicet
decoder_layer_2/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2
decoder_layer_2/stack/1t
decoder_layer_2/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
decoder_layer_2/stack/2t
decoder_layer_2/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
decoder_layer_2/stack/3?
decoder_layer_2/stackPack&decoder_layer_2/strided_slice:output:0 decoder_layer_2/stack/1:output:0 decoder_layer_2/stack/2:output:0 decoder_layer_2/stack/3:output:0*
N*
T0*
_output_shapes
:2
decoder_layer_2/stack?
%decoder_layer_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%decoder_layer_2/strided_slice_1/stack?
'decoder_layer_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'decoder_layer_2/strided_slice_1/stack_1?
'decoder_layer_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'decoder_layer_2/strided_slice_1/stack_2?
decoder_layer_2/strided_slice_1StridedSlicedecoder_layer_2/stack:output:0.decoder_layer_2/strided_slice_1/stack:output:00decoder_layer_2/strided_slice_1/stack_1:output:00decoder_layer_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
decoder_layer_2/strided_slice_1?
/decoder_layer_2/conv2d_transpose/ReadVariableOpReadVariableOp8decoder_layer_2_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype021
/decoder_layer_2/conv2d_transpose/ReadVariableOp?
 decoder_layer_2/conv2d_transposeConv2DBackpropInputdecoder_layer_2/stack:output:07decoder_layer_2/conv2d_transpose/ReadVariableOp:value:0"decoder_layer_1/Relu:activations:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2"
 decoder_layer_2/conv2d_transpose?
&decoder_layer_2/BiasAdd/ReadVariableOpReadVariableOp/decoder_layer_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&decoder_layer_2/BiasAdd/ReadVariableOp?
decoder_layer_2/BiasAddBiasAdd)decoder_layer_2/conv2d_transpose:output:0.decoder_layer_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
decoder_layer_2/BiasAdd?
decoder_layer_2/ReluRelu decoder_layer_2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
decoder_layer_2/Relu?
IdentityIdentity"decoder_layer_2/Relu:activations:0^code/BiasAdd/ReadVariableOp^code/MatMul/ReadVariableOp'^decoder_layer_1/BiasAdd/ReadVariableOp0^decoder_layer_1/conv2d_transpose/ReadVariableOp'^decoder_layer_2/BiasAdd/ReadVariableOp0^decoder_layer_2/conv2d_transpose/ReadVariableOp,^decoder_layer_extra1/BiasAdd/ReadVariableOp5^decoder_layer_extra1/conv2d_transpose/ReadVariableOp,^decoder_layer_extra2/BiasAdd/ReadVariableOp5^decoder_layer_extra2/conv2d_transpose/ReadVariableOp'^encoder_layer_1/BiasAdd/ReadVariableOp&^encoder_layer_1/Conv2D/ReadVariableOp,^encoder_layer_extra1/BiasAdd/ReadVariableOp+^encoder_layer_extra1/Conv2D/ReadVariableOp,^encoder_layer_extra2/BiasAdd/ReadVariableOp+^encoder_layer_extra2/Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:?????????: : : : : : : : : : : : : : : : 2:
code/BiasAdd/ReadVariableOpcode/BiasAdd/ReadVariableOp28
code/MatMul/ReadVariableOpcode/MatMul/ReadVariableOp2P
&decoder_layer_1/BiasAdd/ReadVariableOp&decoder_layer_1/BiasAdd/ReadVariableOp2b
/decoder_layer_1/conv2d_transpose/ReadVariableOp/decoder_layer_1/conv2d_transpose/ReadVariableOp2P
&decoder_layer_2/BiasAdd/ReadVariableOp&decoder_layer_2/BiasAdd/ReadVariableOp2b
/decoder_layer_2/conv2d_transpose/ReadVariableOp/decoder_layer_2/conv2d_transpose/ReadVariableOp2Z
+decoder_layer_extra1/BiasAdd/ReadVariableOp+decoder_layer_extra1/BiasAdd/ReadVariableOp2l
4decoder_layer_extra1/conv2d_transpose/ReadVariableOp4decoder_layer_extra1/conv2d_transpose/ReadVariableOp2Z
+decoder_layer_extra2/BiasAdd/ReadVariableOp+decoder_layer_extra2/BiasAdd/ReadVariableOp2l
4decoder_layer_extra2/conv2d_transpose/ReadVariableOp4decoder_layer_extra2/conv2d_transpose/ReadVariableOp2P
&encoder_layer_1/BiasAdd/ReadVariableOp&encoder_layer_1/BiasAdd/ReadVariableOp2N
%encoder_layer_1/Conv2D/ReadVariableOp%encoder_layer_1/Conv2D/ReadVariableOp2Z
+encoder_layer_extra1/BiasAdd/ReadVariableOp+encoder_layer_extra1/BiasAdd/ReadVariableOp2X
*encoder_layer_extra1/Conv2D/ReadVariableOp*encoder_layer_extra1/Conv2D/ReadVariableOp2Z
+encoder_layer_extra2/BiasAdd/ReadVariableOp+encoder_layer_extra2/BiasAdd/ReadVariableOp2X
*encoder_layer_extra2/Conv2D/ReadVariableOp*encoder_layer_extra2/Conv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
E
)__inference_reshape_1_layer_call_fn_13617

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_reshape_1_layer_call_and_return_conditional_losses_127362
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
4__inference_decoder_layer_extra2_layer_call_fn_12534

inputs!
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_decoder_layer_extra2_layer_call_and_return_conditional_losses_125242
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
/__inference_autoencoder_cnn_layer_call_fn_13010
input_1!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:#
	unknown_3:
	unknown_4:
	unknown_5:	?
	unknown_6:#
	unknown_7:
	unknown_8:#
	unknown_9:

unknown_10:$

unknown_11:

unknown_12:$

unknown_13:

unknown_14:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_autoencoder_cnn_layer_call_and_return_conditional_losses_129382
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:?????????: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_1
?
^
B__inference_flatten_layer_call_and_return_conditional_losses_12704

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
/__inference_autoencoder_cnn_layer_call_fn_13489

inputs!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:#
	unknown_3:
	unknown_4:
	unknown_5:	?
	unknown_6:#
	unknown_7:
	unknown_8:#
	unknown_9:

unknown_10:$

unknown_11:

unknown_12:$

unknown_13:

unknown_14:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_autoencoder_cnn_layer_call_and_return_conditional_losses_129382
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:?????????: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
4__inference_decoder_layer_extra1_layer_call_fn_12489

inputs!
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_decoder_layer_extra1_layer_call_and_return_conditional_losses_124792
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?%
?
J__inference_decoder_layer_2_layer_call_and_return_conditional_losses_12614

inputsB
(conv2d_transpose_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1T
stack/3Const*
_output_shapes
: *
dtype0*
value	B :2	
stack/3?
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_transpose/ReadVariableOp?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingSAME*
strides
2
conv2d_transpose?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?	
?
?__inference_code_layer_call_and_return_conditional_losses_12716

inputs1
matmul_readvariableop_resource:	?-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
/__inference_decoder_layer_1_layer_call_fn_12579

inputs!
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_decoder_layer_1_layer_call_and_return_conditional_losses_125692
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
`
D__inference_reshape_1_layer_call_and_return_conditional_losses_12736

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/2d
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/3?
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapew
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:?????????2	
Reshapel
IdentityIdentityReshape:output:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
O__inference_encoder_layer_extra2_layer_call_and_return_conditional_losses_13539

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?y
?
__inference__traced_save_13811
file_prefix:
6savev2_encoder_layer_extra1_kernel_read_readvariableop8
4savev2_encoder_layer_extra1_bias_read_readvariableop:
6savev2_encoder_layer_extra2_kernel_read_readvariableop8
4savev2_encoder_layer_extra2_bias_read_readvariableop5
1savev2_encoder_layer_1_kernel_read_readvariableop3
/savev2_encoder_layer_1_bias_read_readvariableop*
&savev2_code_kernel_read_readvariableop(
$savev2_code_bias_read_readvariableop:
6savev2_decoder_layer_extra1_kernel_read_readvariableop8
4savev2_decoder_layer_extra1_bias_read_readvariableop:
6savev2_decoder_layer_extra2_kernel_read_readvariableop8
4savev2_decoder_layer_extra2_bias_read_readvariableop5
1savev2_decoder_layer_1_kernel_read_readvariableop3
/savev2_decoder_layer_1_bias_read_readvariableop5
1savev2_decoder_layer_2_kernel_read_readvariableop3
/savev2_decoder_layer_2_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableopA
=savev2_adam_encoder_layer_extra1_kernel_m_read_readvariableop?
;savev2_adam_encoder_layer_extra1_bias_m_read_readvariableopA
=savev2_adam_encoder_layer_extra2_kernel_m_read_readvariableop?
;savev2_adam_encoder_layer_extra2_bias_m_read_readvariableop<
8savev2_adam_encoder_layer_1_kernel_m_read_readvariableop:
6savev2_adam_encoder_layer_1_bias_m_read_readvariableop1
-savev2_adam_code_kernel_m_read_readvariableop/
+savev2_adam_code_bias_m_read_readvariableopA
=savev2_adam_decoder_layer_extra1_kernel_m_read_readvariableop?
;savev2_adam_decoder_layer_extra1_bias_m_read_readvariableopA
=savev2_adam_decoder_layer_extra2_kernel_m_read_readvariableop?
;savev2_adam_decoder_layer_extra2_bias_m_read_readvariableop<
8savev2_adam_decoder_layer_1_kernel_m_read_readvariableop:
6savev2_adam_decoder_layer_1_bias_m_read_readvariableop<
8savev2_adam_decoder_layer_2_kernel_m_read_readvariableop:
6savev2_adam_decoder_layer_2_bias_m_read_readvariableopA
=savev2_adam_encoder_layer_extra1_kernel_v_read_readvariableop?
;savev2_adam_encoder_layer_extra1_bias_v_read_readvariableopA
=savev2_adam_encoder_layer_extra2_kernel_v_read_readvariableop?
;savev2_adam_encoder_layer_extra2_bias_v_read_readvariableop<
8savev2_adam_encoder_layer_1_kernel_v_read_readvariableop:
6savev2_adam_encoder_layer_1_bias_v_read_readvariableop1
-savev2_adam_code_kernel_v_read_readvariableop/
+savev2_adam_code_bias_v_read_readvariableopA
=savev2_adam_decoder_layer_extra1_kernel_v_read_readvariableop?
;savev2_adam_decoder_layer_extra1_bias_v_read_readvariableopA
=savev2_adam_decoder_layer_extra2_kernel_v_read_readvariableop?
;savev2_adam_decoder_layer_extra2_bias_v_read_readvariableop<
8savev2_adam_decoder_layer_1_kernel_v_read_readvariableop:
6savev2_adam_decoder_layer_1_bias_v_read_readvariableop<
8savev2_adam_decoder_layer_2_kernel_v_read_readvariableop:
6savev2_adam_decoder_layer_2_bias_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename? 
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
::*
dtype0*?
value?B?:B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
::*
dtype0*?
value~B|:B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:06savev2_encoder_layer_extra1_kernel_read_readvariableop4savev2_encoder_layer_extra1_bias_read_readvariableop6savev2_encoder_layer_extra2_kernel_read_readvariableop4savev2_encoder_layer_extra2_bias_read_readvariableop1savev2_encoder_layer_1_kernel_read_readvariableop/savev2_encoder_layer_1_bias_read_readvariableop&savev2_code_kernel_read_readvariableop$savev2_code_bias_read_readvariableop6savev2_decoder_layer_extra1_kernel_read_readvariableop4savev2_decoder_layer_extra1_bias_read_readvariableop6savev2_decoder_layer_extra2_kernel_read_readvariableop4savev2_decoder_layer_extra2_bias_read_readvariableop1savev2_decoder_layer_1_kernel_read_readvariableop/savev2_decoder_layer_1_bias_read_readvariableop1savev2_decoder_layer_2_kernel_read_readvariableop/savev2_decoder_layer_2_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop=savev2_adam_encoder_layer_extra1_kernel_m_read_readvariableop;savev2_adam_encoder_layer_extra1_bias_m_read_readvariableop=savev2_adam_encoder_layer_extra2_kernel_m_read_readvariableop;savev2_adam_encoder_layer_extra2_bias_m_read_readvariableop8savev2_adam_encoder_layer_1_kernel_m_read_readvariableop6savev2_adam_encoder_layer_1_bias_m_read_readvariableop-savev2_adam_code_kernel_m_read_readvariableop+savev2_adam_code_bias_m_read_readvariableop=savev2_adam_decoder_layer_extra1_kernel_m_read_readvariableop;savev2_adam_decoder_layer_extra1_bias_m_read_readvariableop=savev2_adam_decoder_layer_extra2_kernel_m_read_readvariableop;savev2_adam_decoder_layer_extra2_bias_m_read_readvariableop8savev2_adam_decoder_layer_1_kernel_m_read_readvariableop6savev2_adam_decoder_layer_1_bias_m_read_readvariableop8savev2_adam_decoder_layer_2_kernel_m_read_readvariableop6savev2_adam_decoder_layer_2_bias_m_read_readvariableop=savev2_adam_encoder_layer_extra1_kernel_v_read_readvariableop;savev2_adam_encoder_layer_extra1_bias_v_read_readvariableop=savev2_adam_encoder_layer_extra2_kernel_v_read_readvariableop;savev2_adam_encoder_layer_extra2_bias_v_read_readvariableop8savev2_adam_encoder_layer_1_kernel_v_read_readvariableop6savev2_adam_encoder_layer_1_bias_v_read_readvariableop-savev2_adam_code_kernel_v_read_readvariableop+savev2_adam_code_bias_v_read_readvariableop=savev2_adam_decoder_layer_extra1_kernel_v_read_readvariableop;savev2_adam_decoder_layer_extra1_bias_v_read_readvariableop=savev2_adam_decoder_layer_extra2_kernel_v_read_readvariableop;savev2_adam_decoder_layer_extra2_bias_v_read_readvariableop8savev2_adam_decoder_layer_1_kernel_v_read_readvariableop6savev2_adam_decoder_layer_1_bias_v_read_readvariableop8savev2_adam_decoder_layer_2_kernel_v_read_readvariableop6savev2_adam_decoder_layer_2_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *H
dtypes>
<2:	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*?
_input_shapes?
?: :::::::	?:::::::::: : : : : : : : : :::::::	?::::::::::::::::	?:::::::::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::%!

_output_shapes
:	?: 

_output_shapes
::,	(
&
_output_shapes
:: 


_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::% !

_output_shapes
:	?: !

_output_shapes
::,"(
&
_output_shapes
:: #

_output_shapes
::,$(
&
_output_shapes
:: %

_output_shapes
::,&(
&
_output_shapes
:: '

_output_shapes
::,((
&
_output_shapes
:: )

_output_shapes
::,*(
&
_output_shapes
:: +

_output_shapes
::,,(
&
_output_shapes
:: -

_output_shapes
::,.(
&
_output_shapes
:: /

_output_shapes
::%0!

_output_shapes
:	?: 1

_output_shapes
::,2(
&
_output_shapes
:: 3

_output_shapes
::,4(
&
_output_shapes
:: 5

_output_shapes
::,6(
&
_output_shapes
:: 7

_output_shapes
::,8(
&
_output_shapes
:: 9

_output_shapes
:::

_output_shapes
: "?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
C
input_18
serving_default_input_1:0?????????K
decoder_layer_28
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?p
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer-4
layer_with_weights-3
layer-5
layer-6
layer_with_weights-4
layer-7
	layer_with_weights-5
	layer-8

layer_with_weights-6

layer-9
layer_with_weights-7
layer-10
	optimizer
trainable_variables
	variables
regularization_losses
	keras_api

signatures
+?&call_and_return_all_conditional_losses
?__call__
?_default_save_signature"?l
_tf_keras_sequential?l{"name": "autoencoder_cnn", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Sequential", "config": {"name": "autoencoder_cnn", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 8, 8, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}}, {"class_name": "Reshape", "config": {"name": "reshape", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [8, 8, 1]}}}, {"class_name": "Conv2D", "config": {"name": "encoder_layer_extra1", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "encoder_layer_extra2", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "encoder_layer_1", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "code", "trainable": true, "dtype": "float32", "units": 8, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Reshape", "config": {"name": "reshape_1", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [2, 2, 2]}}}, {"class_name": "Conv2DTranspose", "config": {"name": "decoder_layer_extra1", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}}, {"class_name": "Conv2DTranspose", "config": {"name": "decoder_layer_extra2", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}}, {"class_name": "Conv2DTranspose", "config": {"name": "decoder_layer_1", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}}, {"class_name": "Conv2DTranspose", "config": {"name": "decoder_layer_2", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}}]}, "shared_object_id": 28, "build_input_shape": {"class_name": "TensorShape", "items": [null, 8, 8, 1]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 8, 8, 1]}, "float32", "input_1"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "autoencoder_cnn", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 8, 8, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "shared_object_id": 0}, {"class_name": "Reshape", "config": {"name": "reshape", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [8, 8, 1]}}, "shared_object_id": 1}, {"class_name": "Conv2D", "config": {"name": "encoder_layer_extra1", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 2}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 3}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 4}, {"class_name": "Conv2D", "config": {"name": "encoder_layer_extra2", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 5}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 6}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 7}, {"class_name": "Conv2D", "config": {"name": "encoder_layer_1", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 8}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 9}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 10}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "shared_object_id": 11}, {"class_name": "Dense", "config": {"name": "code", "trainable": true, "dtype": "float32", "units": 8, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 12}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 13}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 14}, {"class_name": "Reshape", "config": {"name": "reshape_1", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [2, 2, 2]}}, "shared_object_id": 15}, {"class_name": "Conv2DTranspose", "config": {"name": "decoder_layer_extra1", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 16}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 17}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "shared_object_id": 18}, {"class_name": "Conv2DTranspose", "config": {"name": "decoder_layer_extra2", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 19}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 20}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "shared_object_id": 21}, {"class_name": "Conv2DTranspose", "config": {"name": "decoder_layer_1", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 22}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 23}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "shared_object_id": 24}, {"class_name": "Conv2DTranspose", "config": {"name": "decoder_layer_2", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 25}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 26}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "shared_object_id": 27}]}}, "training_config": {"loss": {"class_name": "MeanSquaredError", "config": {"reduction": "auto", "name": "mean_squared_error"}, "shared_object_id": 29}, "metrics": [[{"class_name": "MeanSquaredError", "config": {"name": "mean_squared_error", "dtype": "float32"}, "shared_object_id": 30}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?
trainable_variables
	variables
regularization_losses
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "reshape", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Reshape", "config": {"name": "reshape", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [8, 8, 1]}}, "shared_object_id": 1}
?


kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?	
_tf_keras_layer?	{"name": "encoder_layer_extra1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "encoder_layer_extra1", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 2}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 3}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 4, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 1}}, "shared_object_id": 31}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 8, 8, 1]}}
?


kernel
bias
trainable_variables
	variables
 regularization_losses
!	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?	
_tf_keras_layer?	{"name": "encoder_layer_extra2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "encoder_layer_extra2", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 5}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 6}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 7, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 16}}, "shared_object_id": 32}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 8, 8, 16]}}
?


"kernel
#bias
$trainable_variables
%	variables
&regularization_losses
'	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?	
_tf_keras_layer?	{"name": "encoder_layer_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "encoder_layer_1", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 8}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 9}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 10, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 16}}, "shared_object_id": 33}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 8, 8, 16]}}
?
(trainable_variables
)	variables
*regularization_losses
+	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "shared_object_id": 11, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}, "shared_object_id": 34}}
?

,kernel
-bias
.trainable_variables
/	variables
0regularization_losses
1	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "code", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "code", "trainable": true, "dtype": "float32", "units": 8, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 12}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 13}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 14, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}, "shared_object_id": 35}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}
?
2trainable_variables
3	variables
4regularization_losses
5	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "reshape_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Reshape", "config": {"name": "reshape_1", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [2, 2, 2]}}, "shared_object_id": 15}
?

6kernel
7bias
8trainable_variables
9	variables
:regularization_losses
;	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?	
_tf_keras_layer?	{"name": "decoder_layer_extra1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2DTranspose", "config": {"name": "decoder_layer_extra1", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 16}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 17}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "shared_object_id": 18, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 2}}, "shared_object_id": 36}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 2, 2, 2]}}
?

<kernel
=bias
>trainable_variables
?	variables
@regularization_losses
A	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?	
_tf_keras_layer?	{"name": "decoder_layer_extra2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2DTranspose", "config": {"name": "decoder_layer_extra2", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 19}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 20}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "shared_object_id": 21, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 16}}, "shared_object_id": 37}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 2, 2, 16]}}
?

Bkernel
Cbias
Dtrainable_variables
E	variables
Fregularization_losses
G	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?	
_tf_keras_layer?	{"name": "decoder_layer_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2DTranspose", "config": {"name": "decoder_layer_1", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 22}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 23}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "shared_object_id": 24, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 16}}, "shared_object_id": 38}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 2, 2, 16]}}
?

Hkernel
Ibias
Jtrainable_variables
K	variables
Lregularization_losses
M	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?	
_tf_keras_layer?	{"name": "decoder_layer_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2DTranspose", "config": {"name": "decoder_layer_2", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 25}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 26}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "shared_object_id": 27, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 16}}, "shared_object_id": 39}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 4, 4, 16]}}
?
Niter

Obeta_1

Pbeta_2
	Qdecay
Rlearning_ratem?m?m?m?"m?#m?,m?-m?6m?7m?<m?=m?Bm?Cm?Hm?Im?v?v?v?v?"v?#v?,v?-v?6v?7v?<v?=v?Bv?Cv?Hv?Iv?"
	optimizer
?
0
1
2
3
"4
#5
,6
-7
68
79
<10
=11
B12
C13
H14
I15"
trackable_list_wrapper
?
0
1
2
3
"4
#5
,6
-7
68
79
<10
=11
B12
C13
H14
I15"
trackable_list_wrapper
 "
trackable_list_wrapper
?
trainable_variables
	variables
Slayer_metrics
Tlayer_regularization_losses
Unon_trainable_variables
Vmetrics
regularization_losses

Wlayers
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
trainable_variables
Xlayer_metrics
Ylayer_regularization_losses
	variables
Znon_trainable_variables
[metrics
regularization_losses

\layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
5:32encoder_layer_extra1/kernel
':%2encoder_layer_extra1/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
trainable_variables
]layer_metrics
^layer_regularization_losses
	variables
_non_trainable_variables
`metrics
regularization_losses

alayers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
5:32encoder_layer_extra2/kernel
':%2encoder_layer_extra2/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
trainable_variables
blayer_metrics
clayer_regularization_losses
	variables
dnon_trainable_variables
emetrics
 regularization_losses

flayers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
0:.2encoder_layer_1/kernel
": 2encoder_layer_1/bias
.
"0
#1"
trackable_list_wrapper
.
"0
#1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
$trainable_variables
glayer_metrics
hlayer_regularization_losses
%	variables
inon_trainable_variables
jmetrics
&regularization_losses

klayers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
(trainable_variables
llayer_metrics
mlayer_regularization_losses
)	variables
nnon_trainable_variables
ometrics
*regularization_losses

players
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	?2code/kernel
:2	code/bias
.
,0
-1"
trackable_list_wrapper
.
,0
-1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
.trainable_variables
qlayer_metrics
rlayer_regularization_losses
/	variables
snon_trainable_variables
tmetrics
0regularization_losses

ulayers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
2trainable_variables
vlayer_metrics
wlayer_regularization_losses
3	variables
xnon_trainable_variables
ymetrics
4regularization_losses

zlayers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
5:32decoder_layer_extra1/kernel
':%2decoder_layer_extra1/bias
.
60
71"
trackable_list_wrapper
.
60
71"
trackable_list_wrapper
 "
trackable_list_wrapper
?
8trainable_variables
{layer_metrics
|layer_regularization_losses
9	variables
}non_trainable_variables
~metrics
:regularization_losses

layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
5:32decoder_layer_extra2/kernel
':%2decoder_layer_extra2/bias
.
<0
=1"
trackable_list_wrapper
.
<0
=1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
>trainable_variables
?layer_metrics
 ?layer_regularization_losses
?	variables
?non_trainable_variables
?metrics
@regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
0:.2decoder_layer_1/kernel
": 2decoder_layer_1/bias
.
B0
C1"
trackable_list_wrapper
.
B0
C1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Dtrainable_variables
?layer_metrics
 ?layer_regularization_losses
E	variables
?non_trainable_variables
?metrics
Fregularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
0:.2decoder_layer_2/kernel
": 2decoder_layer_2/bias
.
H0
I1"
trackable_list_wrapper
.
H0
I1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Jtrainable_variables
?layer_metrics
 ?layer_regularization_losses
K	variables
?non_trainable_variables
?metrics
Lregularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
n
0
1
2
3
4
5
6
7
	8

9
10"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?

?total

?count
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}, "shared_object_id": 40}
?

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "MeanSquaredError", "name": "mean_squared_error", "dtype": "float32", "config": {"name": "mean_squared_error", "dtype": "float32"}, "shared_object_id": 30}
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
::82"Adam/encoder_layer_extra1/kernel/m
,:*2 Adam/encoder_layer_extra1/bias/m
::82"Adam/encoder_layer_extra2/kernel/m
,:*2 Adam/encoder_layer_extra2/bias/m
5:32Adam/encoder_layer_1/kernel/m
':%2Adam/encoder_layer_1/bias/m
#:!	?2Adam/code/kernel/m
:2Adam/code/bias/m
::82"Adam/decoder_layer_extra1/kernel/m
,:*2 Adam/decoder_layer_extra1/bias/m
::82"Adam/decoder_layer_extra2/kernel/m
,:*2 Adam/decoder_layer_extra2/bias/m
5:32Adam/decoder_layer_1/kernel/m
':%2Adam/decoder_layer_1/bias/m
5:32Adam/decoder_layer_2/kernel/m
':%2Adam/decoder_layer_2/bias/m
::82"Adam/encoder_layer_extra1/kernel/v
,:*2 Adam/encoder_layer_extra1/bias/v
::82"Adam/encoder_layer_extra2/kernel/v
,:*2 Adam/encoder_layer_extra2/bias/v
5:32Adam/encoder_layer_1/kernel/v
':%2Adam/encoder_layer_1/bias/v
#:!	?2Adam/code/kernel/v
:2Adam/code/bias/v
::82"Adam/decoder_layer_extra1/kernel/v
,:*2 Adam/decoder_layer_extra1/bias/v
::82"Adam/decoder_layer_extra2/kernel/v
,:*2 Adam/decoder_layer_extra2/bias/v
5:32Adam/decoder_layer_1/kernel/v
':%2Adam/decoder_layer_1/bias/v
5:32Adam/decoder_layer_2/kernel/v
':%2Adam/decoder_layer_2/bias/v
?2?
J__inference_autoencoder_cnn_layer_call_and_return_conditional_losses_13282
J__inference_autoencoder_cnn_layer_call_and_return_conditional_losses_13415
J__inference_autoencoder_cnn_layer_call_and_return_conditional_losses_13057
J__inference_autoencoder_cnn_layer_call_and_return_conditional_losses_13104?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
/__inference_autoencoder_cnn_layer_call_fn_12794
/__inference_autoencoder_cnn_layer_call_fn_13452
/__inference_autoencoder_cnn_layer_call_fn_13489
/__inference_autoencoder_cnn_layer_call_fn_13010?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
 __inference__wrapped_model_12444?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *.?+
)?&
input_1?????????
?2?
B__inference_reshape_layer_call_and_return_conditional_losses_13503?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_reshape_layer_call_fn_13508?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
O__inference_encoder_layer_extra1_layer_call_and_return_conditional_losses_13519?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
4__inference_encoder_layer_extra1_layer_call_fn_13528?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
O__inference_encoder_layer_extra2_layer_call_and_return_conditional_losses_13539?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
4__inference_encoder_layer_extra2_layer_call_fn_13548?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
J__inference_encoder_layer_1_layer_call_and_return_conditional_losses_13559?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
/__inference_encoder_layer_1_layer_call_fn_13568?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_flatten_layer_call_and_return_conditional_losses_13574?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_flatten_layer_call_fn_13579?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
?__inference_code_layer_call_and_return_conditional_losses_13589?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
$__inference_code_layer_call_fn_13598?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_reshape_1_layer_call_and_return_conditional_losses_13612?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_reshape_1_layer_call_fn_13617?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
O__inference_decoder_layer_extra1_layer_call_and_return_conditional_losses_12479?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+???????????????????????????
?2?
4__inference_decoder_layer_extra1_layer_call_fn_12489?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+???????????????????????????
?2?
O__inference_decoder_layer_extra2_layer_call_and_return_conditional_losses_12524?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+???????????????????????????
?2?
4__inference_decoder_layer_extra2_layer_call_fn_12534?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+???????????????????????????
?2?
J__inference_decoder_layer_1_layer_call_and_return_conditional_losses_12569?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+???????????????????????????
?2?
/__inference_decoder_layer_1_layer_call_fn_12579?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+???????????????????????????
?2?
J__inference_decoder_layer_2_layer_call_and_return_conditional_losses_12614?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+???????????????????????????
?2?
/__inference_decoder_layer_2_layer_call_fn_12624?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+???????????????????????????
?B?
#__inference_signature_wrapper_13149input_1"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
 __inference__wrapped_model_12444?"#,-67<=BCHI8?5
.?+
)?&
input_1?????????
? "I?F
D
decoder_layer_21?.
decoder_layer_2??????????
J__inference_autoencoder_cnn_layer_call_and_return_conditional_losses_13057?"#,-67<=BCHI@?=
6?3
)?&
input_1?????????
p 

 
? "??<
5?2
0+???????????????????????????
? ?
J__inference_autoencoder_cnn_layer_call_and_return_conditional_losses_13104?"#,-67<=BCHI@?=
6?3
)?&
input_1?????????
p

 
? "??<
5?2
0+???????????????????????????
? ?
J__inference_autoencoder_cnn_layer_call_and_return_conditional_losses_13282?"#,-67<=BCHI??<
5?2
(?%
inputs?????????
p 

 
? "-?*
#? 
0?????????
? ?
J__inference_autoencoder_cnn_layer_call_and_return_conditional_losses_13415?"#,-67<=BCHI??<
5?2
(?%
inputs?????????
p

 
? "-?*
#? 
0?????????
? ?
/__inference_autoencoder_cnn_layer_call_fn_12794?"#,-67<=BCHI@?=
6?3
)?&
input_1?????????
p 

 
? "2?/+????????????????????????????
/__inference_autoencoder_cnn_layer_call_fn_13010?"#,-67<=BCHI@?=
6?3
)?&
input_1?????????
p

 
? "2?/+????????????????????????????
/__inference_autoencoder_cnn_layer_call_fn_13452?"#,-67<=BCHI??<
5?2
(?%
inputs?????????
p 

 
? "2?/+????????????????????????????
/__inference_autoencoder_cnn_layer_call_fn_13489?"#,-67<=BCHI??<
5?2
(?%
inputs?????????
p

 
? "2?/+????????????????????????????
?__inference_code_layer_call_and_return_conditional_losses_13589],-0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????
? x
$__inference_code_layer_call_fn_13598P,-0?-
&?#
!?
inputs??????????
? "???????????
J__inference_decoder_layer_1_layer_call_and_return_conditional_losses_12569?BCI?F
??<
:?7
inputs+???????????????????????????
? "??<
5?2
0+???????????????????????????
? ?
/__inference_decoder_layer_1_layer_call_fn_12579?BCI?F
??<
:?7
inputs+???????????????????????????
? "2?/+????????????????????????????
J__inference_decoder_layer_2_layer_call_and_return_conditional_losses_12614?HII?F
??<
:?7
inputs+???????????????????????????
? "??<
5?2
0+???????????????????????????
? ?
/__inference_decoder_layer_2_layer_call_fn_12624?HII?F
??<
:?7
inputs+???????????????????????????
? "2?/+????????????????????????????
O__inference_decoder_layer_extra1_layer_call_and_return_conditional_losses_12479?67I?F
??<
:?7
inputs+???????????????????????????
? "??<
5?2
0+???????????????????????????
? ?
4__inference_decoder_layer_extra1_layer_call_fn_12489?67I?F
??<
:?7
inputs+???????????????????????????
? "2?/+????????????????????????????
O__inference_decoder_layer_extra2_layer_call_and_return_conditional_losses_12524?<=I?F
??<
:?7
inputs+???????????????????????????
? "??<
5?2
0+???????????????????????????
? ?
4__inference_decoder_layer_extra2_layer_call_fn_12534?<=I?F
??<
:?7
inputs+???????????????????????????
? "2?/+????????????????????????????
J__inference_encoder_layer_1_layer_call_and_return_conditional_losses_13559l"#7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????
? ?
/__inference_encoder_layer_1_layer_call_fn_13568_"#7?4
-?*
(?%
inputs?????????
? " ???????????
O__inference_encoder_layer_extra1_layer_call_and_return_conditional_losses_13519l7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????
? ?
4__inference_encoder_layer_extra1_layer_call_fn_13528_7?4
-?*
(?%
inputs?????????
? " ???????????
O__inference_encoder_layer_extra2_layer_call_and_return_conditional_losses_13539l7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????
? ?
4__inference_encoder_layer_extra2_layer_call_fn_13548_7?4
-?*
(?%
inputs?????????
? " ???????????
B__inference_flatten_layer_call_and_return_conditional_losses_13574a7?4
-?*
(?%
inputs?????????
? "&?#
?
0??????????
? 
'__inference_flatten_layer_call_fn_13579T7?4
-?*
(?%
inputs?????????
? "????????????
D__inference_reshape_1_layer_call_and_return_conditional_losses_13612`/?,
%?"
 ?
inputs?????????
? "-?*
#? 
0?????????
? ?
)__inference_reshape_1_layer_call_fn_13617S/?,
%?"
 ?
inputs?????????
? " ???????????
B__inference_reshape_layer_call_and_return_conditional_losses_13503h7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????
? ?
'__inference_reshape_layer_call_fn_13508[7?4
-?*
(?%
inputs?????????
? " ???????????
#__inference_signature_wrapper_13149?"#,-67<=BCHIC?@
? 
9?6
4
input_1)?&
input_1?????????"I?F
D
decoder_layer_21?.
decoder_layer_2?????????