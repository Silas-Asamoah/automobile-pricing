Ēń
ęŹ
W
AddN
inputs"T*N
sum"T"
Nint(0"!
Ttype:
2	
B
AssignVariableOp
resource
value"dtype"
dtypetype
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
8
Const
output"dtype"
valuetensor"
dtypetype
h
Equal
x"T
y"T
z
"
Ttype:
2	
"$
incompatible_shape_errorbool(
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
.
Identity

input"T
output"T"	
Ttype
ŗ
If
cond"Tcond
input2Tin
output2Tout"
Tcondtype"
Tin
list(type)("
Tout
list(type)("
then_branchfunc"
else_branchfunc" 
output_shapeslist(shape)
 
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
delete_old_dirsbool(
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
X
PlaceholderWithDefault
input"dtype
output"dtype"
dtypetype"
shapeshape
@
ReadVariableOp
resource
value"dtype"
dtypetype
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
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
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
P
ScalarSummary
tags
values"T
summary"
Ttype:
2	
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
ö
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
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape
9
VarIsInitializedOp
resource
is_initialized
"serve*2.0.02unknown8¤Ī

global_step/Initializer/zerosConst*
_class
loc:@global_step*
value	B	 R *
dtype0	*
_output_shapes
: 

global_stepVarHandleOp*
shape: *
shared_nameglobal_step*
_class
loc:@global_step*
dtype0	*
_output_shapes
: 
g
,global_step/IsInitialized/VarIsInitializedOpVarIsInitializedOpglobal_step*
_output_shapes
: 
_
global_step/AssignAssignVariableOpglobal_stepglobal_step/Initializer/zeros*
dtype0	
c
global_step/Read/ReadVariableOpReadVariableOpglobal_step*
dtype0	*
_output_shapes
: 
f
PlaceholderPlaceholder*
shape:’’’’’’’’’*
dtype0*#
_output_shapes
:’’’’’’’’’
h
Placeholder_1Placeholder*
shape:’’’’’’’’’*
dtype0*#
_output_shapes
:’’’’’’’’’
j
linear/linear_model/CastCastPlaceholder*

SrcT0*

DstT0*#
_output_shapes
:’’’’’’’’’
n
linear/linear_model/Cast_1CastPlaceholder_1*

SrcT0*

DstT0*#
_output_shapes
:’’’’’’’’’
Ź
9linear/linear_model/curb-weight/weights/Initializer/zerosConst*:
_class0
.,loc:@linear/linear_model/curb-weight/weights*
valueB*    *
dtype0*
_output_shapes

:
ę
'linear/linear_model/curb-weight/weightsVarHandleOp*
shape
:*8
shared_name)'linear/linear_model/curb-weight/weights*:
_class0
.,loc:@linear/linear_model/curb-weight/weights*
dtype0*
_output_shapes
: 

Hlinear/linear_model/curb-weight/weights/IsInitialized/VarIsInitializedOpVarIsInitializedOp'linear/linear_model/curb-weight/weights*
_output_shapes
: 
³
.linear/linear_model/curb-weight/weights/AssignAssignVariableOp'linear/linear_model/curb-weight/weights9linear/linear_model/curb-weight/weights/Initializer/zeros*
dtype0
£
;linear/linear_model/curb-weight/weights/Read/ReadVariableOpReadVariableOp'linear/linear_model/curb-weight/weights*
dtype0*
_output_shapes

:
Ź
9linear/linear_model/highway-mpg/weights/Initializer/zerosConst*:
_class0
.,loc:@linear/linear_model/highway-mpg/weights*
valueB*    *
dtype0*
_output_shapes

:
ę
'linear/linear_model/highway-mpg/weightsVarHandleOp*
shape
:*8
shared_name)'linear/linear_model/highway-mpg/weights*:
_class0
.,loc:@linear/linear_model/highway-mpg/weights*
dtype0*
_output_shapes
: 

Hlinear/linear_model/highway-mpg/weights/IsInitialized/VarIsInitializedOpVarIsInitializedOp'linear/linear_model/highway-mpg/weights*
_output_shapes
: 
³
.linear/linear_model/highway-mpg/weights/AssignAssignVariableOp'linear/linear_model/highway-mpg/weights9linear/linear_model/highway-mpg/weights/Initializer/zeros*
dtype0
£
;linear/linear_model/highway-mpg/weights/Read/ReadVariableOpReadVariableOp'linear/linear_model/highway-mpg/weights*
dtype0*
_output_shapes

:
“
2linear/linear_model/bias_weights/Initializer/zerosConst*3
_class)
'%loc:@linear/linear_model/bias_weights*
valueB*    *
dtype0*
_output_shapes
:
Ķ
 linear/linear_model/bias_weightsVarHandleOp*
shape:*1
shared_name" linear/linear_model/bias_weights*3
_class)
'%loc:@linear/linear_model/bias_weights*
dtype0*
_output_shapes
: 

Alinear/linear_model/bias_weights/IsInitialized/VarIsInitializedOpVarIsInitializedOp linear/linear_model/bias_weights*
_output_shapes
: 

'linear/linear_model/bias_weights/AssignAssignVariableOp linear/linear_model/bias_weights2linear/linear_model/bias_weights/Initializer/zeros*
dtype0

4linear/linear_model/bias_weights/Read/ReadVariableOpReadVariableOp linear/linear_model/bias_weights*
dtype0*
_output_shapes
:
”
Vlinear/linear_model/linear/linear_model/linear/linear_model/curb-weight/ExpandDims/dimConst*
valueB :
’’’’’’’’’*
dtype0*
_output_shapes
: 

Rlinear/linear_model/linear/linear_model/linear/linear_model/curb-weight/ExpandDims
ExpandDimslinear/linear_model/CastVlinear/linear_model/linear/linear_model/linear/linear_model/curb-weight/ExpandDims/dim*
T0*'
_output_shapes
:’’’’’’’’’
Ļ
Mlinear/linear_model/linear/linear_model/linear/linear_model/curb-weight/ShapeShapeRlinear/linear_model/linear/linear_model/linear/linear_model/curb-weight/ExpandDims*
T0*
_output_shapes
:
„
[linear/linear_model/linear/linear_model/linear/linear_model/curb-weight/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
§
]linear/linear_model/linear/linear_model/linear/linear_model/curb-weight/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
§
]linear/linear_model/linear/linear_model/linear/linear_model/curb-weight/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:

Ulinear/linear_model/linear/linear_model/linear/linear_model/curb-weight/strided_sliceStridedSliceMlinear/linear_model/linear/linear_model/linear/linear_model/curb-weight/Shape[linear/linear_model/linear/linear_model/linear/linear_model/curb-weight/strided_slice/stack]linear/linear_model/linear/linear_model/linear/linear_model/curb-weight/strided_slice/stack_1]linear/linear_model/linear/linear_model/linear/linear_model/curb-weight/strided_slice/stack_2*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: 

Wlinear/linear_model/linear/linear_model/linear/linear_model/curb-weight/Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 
»
Ulinear/linear_model/linear/linear_model/linear/linear_model/curb-weight/Reshape/shapePackUlinear/linear_model/linear/linear_model/linear/linear_model/curb-weight/strided_sliceWlinear/linear_model/linear/linear_model/linear/linear_model/curb-weight/Reshape/shape/1*
T0*
N*
_output_shapes
:
·
Olinear/linear_model/linear/linear_model/linear/linear_model/curb-weight/ReshapeReshapeRlinear/linear_model/linear/linear_model/linear/linear_model/curb-weight/ExpandDimsUlinear/linear_model/linear/linear_model/linear/linear_model/curb-weight/Reshape/shape*
T0*'
_output_shapes
:’’’’’’’’’
Ė
clinear/linear_model/linear/linear_model/linear/linear_model/curb-weight/weighted_sum/ReadVariableOpReadVariableOp'linear/linear_model/curb-weight/weights*
dtype0*
_output_shapes

:
Ę
Tlinear/linear_model/linear/linear_model/linear/linear_model/curb-weight/weighted_sumMatMulOlinear/linear_model/linear/linear_model/linear/linear_model/curb-weight/Reshapeclinear/linear_model/linear/linear_model/linear/linear_model/curb-weight/weighted_sum/ReadVariableOp*
T0*'
_output_shapes
:’’’’’’’’’
”
Vlinear/linear_model/linear/linear_model/linear/linear_model/highway-mpg/ExpandDims/dimConst*
valueB :
’’’’’’’’’*
dtype0*
_output_shapes
: 

Rlinear/linear_model/linear/linear_model/linear/linear_model/highway-mpg/ExpandDims
ExpandDimslinear/linear_model/Cast_1Vlinear/linear_model/linear/linear_model/linear/linear_model/highway-mpg/ExpandDims/dim*
T0*'
_output_shapes
:’’’’’’’’’
Ļ
Mlinear/linear_model/linear/linear_model/linear/linear_model/highway-mpg/ShapeShapeRlinear/linear_model/linear/linear_model/linear/linear_model/highway-mpg/ExpandDims*
T0*
_output_shapes
:
„
[linear/linear_model/linear/linear_model/linear/linear_model/highway-mpg/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
§
]linear/linear_model/linear/linear_model/linear/linear_model/highway-mpg/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
§
]linear/linear_model/linear/linear_model/linear/linear_model/highway-mpg/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:

Ulinear/linear_model/linear/linear_model/linear/linear_model/highway-mpg/strided_sliceStridedSliceMlinear/linear_model/linear/linear_model/linear/linear_model/highway-mpg/Shape[linear/linear_model/linear/linear_model/linear/linear_model/highway-mpg/strided_slice/stack]linear/linear_model/linear/linear_model/linear/linear_model/highway-mpg/strided_slice/stack_1]linear/linear_model/linear/linear_model/linear/linear_model/highway-mpg/strided_slice/stack_2*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: 

Wlinear/linear_model/linear/linear_model/linear/linear_model/highway-mpg/Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 
»
Ulinear/linear_model/linear/linear_model/linear/linear_model/highway-mpg/Reshape/shapePackUlinear/linear_model/linear/linear_model/linear/linear_model/highway-mpg/strided_sliceWlinear/linear_model/linear/linear_model/linear/linear_model/highway-mpg/Reshape/shape/1*
T0*
N*
_output_shapes
:
·
Olinear/linear_model/linear/linear_model/linear/linear_model/highway-mpg/ReshapeReshapeRlinear/linear_model/linear/linear_model/linear/linear_model/highway-mpg/ExpandDimsUlinear/linear_model/linear/linear_model/linear/linear_model/highway-mpg/Reshape/shape*
T0*'
_output_shapes
:’’’’’’’’’
Ė
clinear/linear_model/linear/linear_model/linear/linear_model/highway-mpg/weighted_sum/ReadVariableOpReadVariableOp'linear/linear_model/highway-mpg/weights*
dtype0*
_output_shapes

:
Ę
Tlinear/linear_model/linear/linear_model/linear/linear_model/highway-mpg/weighted_sumMatMulOlinear/linear_model/linear/linear_model/linear/linear_model/highway-mpg/Reshapeclinear/linear_model/linear/linear_model/linear/linear_model/highway-mpg/weighted_sum/ReadVariableOp*
T0*'
_output_shapes
:’’’’’’’’’
æ
Plinear/linear_model/linear/linear_model/linear/linear_model/weighted_sum_no_biasAddNTlinear/linear_model/linear/linear_model/linear/linear_model/curb-weight/weighted_sumTlinear/linear_model/linear/linear_model/linear/linear_model/highway-mpg/weighted_sum*
T0*
N*'
_output_shapes
:’’’’’’’’’
“
Wlinear/linear_model/linear/linear_model/linear/linear_model/weighted_sum/ReadVariableOpReadVariableOp linear/linear_model/bias_weights*
dtype0*
_output_shapes
:
°
Hlinear/linear_model/linear/linear_model/linear/linear_model/weighted_sumBiasAddPlinear/linear_model/linear/linear_model/linear/linear_model/weighted_sum_no_biasWlinear/linear_model/linear/linear_model/linear/linear_model/weighted_sum/ReadVariableOp*
T0*'
_output_shapes
:’’’’’’’’’
k
ReadVariableOpReadVariableOp linear/linear_model/bias_weights*
dtype0*
_output_shapes
:
]
strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
_
strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
_
strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
¶
strided_sliceStridedSliceReadVariableOpstrided_slice/stackstrided_slice/stack_1strided_slice/stack_2*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: 
N
	bias/tagsConst*
valueB
 Bbias*
dtype0*
_output_shapes
: 
P
biasScalarSummary	bias/tagsstrided_slice*
T0*
_output_shapes
: 

,zero_fraction/total_size/Size/ReadVariableOpReadVariableOp'linear/linear_model/curb-weight/weights*
dtype0*
_output_shapes

:
_
zero_fraction/total_size/SizeConst*
value	B	 R*
dtype0	*
_output_shapes
: 

.zero_fraction/total_size/Size_1/ReadVariableOpReadVariableOp'linear/linear_model/highway-mpg/weights*
dtype0*
_output_shapes

:
a
zero_fraction/total_size/Size_1Const*
value	B	 R*
dtype0	*
_output_shapes
: 

zero_fraction/total_size/AddNAddNzero_fraction/total_size/Sizezero_fraction/total_size/Size_1*
T0	*
N*
_output_shapes
: 
`
zero_fraction/total_zero/ConstConst*
value	B	 R *
dtype0	*
_output_shapes
: 

zero_fraction/total_zero/EqualEqualzero_fraction/total_size/Sizezero_fraction/total_zero/Const*
T0	*
_output_shapes
: 

#zero_fraction/total_zero/zero_countIfzero_fraction/total_zero/Equal'linear/linear_model/curb-weight/weightszero_fraction/total_size/Size*B
else_branch3R1
/zero_fraction_total_zero_zero_count_false_10450*
output_shapes
: *
_lower_using_switch_merge(*
Tout
2*
Tcond0
*A
then_branch2R0
.zero_fraction_total_zero_zero_count_true_10449*
Tin
2	*
_output_shapes
: 
~
,zero_fraction/total_zero/zero_count/IdentityIdentity#zero_fraction/total_zero/zero_count*
T0*
_output_shapes
: 
b
 zero_fraction/total_zero/Const_1Const*
value	B	 R *
dtype0	*
_output_shapes
: 

 zero_fraction/total_zero/Equal_1Equalzero_fraction/total_size/Size_1 zero_fraction/total_zero/Const_1*
T0	*
_output_shapes
: 

%zero_fraction/total_zero/zero_count_1If zero_fraction/total_zero/Equal_1'linear/linear_model/highway-mpg/weightszero_fraction/total_size/Size_1*D
else_branch5R3
1zero_fraction_total_zero_zero_count_1_false_10493*
output_shapes
: *
_lower_using_switch_merge(*
Tout
2*
Tcond0
*C
then_branch4R2
0zero_fraction_total_zero_zero_count_1_true_10492*
Tin
2	*
_output_shapes
: 

.zero_fraction/total_zero/zero_count_1/IdentityIdentity%zero_fraction/total_zero/zero_count_1*
T0*
_output_shapes
: 
­
zero_fraction/total_zero/AddNAddN,zero_fraction/total_zero/zero_count/Identity.zero_fraction/total_zero/zero_count_1/Identity*
T0*
N*
_output_shapes
: 
y
"zero_fraction/compute/float32_sizeCastzero_fraction/total_size/AddN*

SrcT0	*

DstT0*
_output_shapes
: 

zero_fraction/compute/truedivRealDivzero_fraction/total_zero/AddN"zero_fraction/compute/float32_size*
T0*
_output_shapes
: 
n
"zero_fraction/zero_fraction_or_nanIdentityzero_fraction/compute/truediv*
T0*
_output_shapes
: 
v
fraction_of_zero_weights/tagsConst*)
value B Bfraction_of_zero_weights*
dtype0*
_output_shapes
: 

fraction_of_zero_weightsScalarSummaryfraction_of_zero_weights/tags"zero_fraction/zero_fraction_or_nan*
T0*
_output_shapes
: 

head/logits/ShapeShapeHlinear/linear_model/linear/linear_model/linear/linear_model/weighted_sum*
T0*
_output_shapes
:
g
%head/logits/assert_rank_at_least/rankConst*
value	B :*
dtype0*
_output_shapes
: 
W
Ohead/logits/assert_rank_at_least/assert_type/statically_determined_correct_typeNoOp
H
@head/logits/assert_rank_at_least/static_checks_determined_all_okNoOp

initNoOp

init_all_tablesNoOp

init_1NoOp
4

group_depsNoOp^init^init_1^init_all_tables
Y
save/filename/inputConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
n
save/filenamePlaceholderWithDefaultsave/filename/input*
shape: *
dtype0*
_output_shapes
: 
e

save/ConstPlaceholderWithDefaultsave/filename*
shape: *
dtype0*
_output_shapes
: 

save/StringJoin/inputs_1Const*<
value3B1 B+_temp_7968f7e710de46bfa43092aa1fa36f3e/part*
dtype0*
_output_shapes
: 
d
save/StringJoin
StringJoin
save/Constsave/StringJoin/inputs_1*
N*
_output_shapes
: 
Q
save/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
k
save/ShardedFilename/shardConst"/device:CPU:0*
value	B : *
dtype0*
_output_shapes
: 

save/ShardedFilenameShardedFilenamesave/StringJoinsave/ShardedFilename/shardsave/num_shards"/device:CPU:0*
_output_shapes
: 
ņ
save/SaveV2/tensor_namesConst"/device:CPU:0*
valueBBglobal_stepB linear/linear_model/bias_weightsB'linear/linear_model/curb-weight/weightsB'linear/linear_model/highway-mpg/weights*
dtype0*
_output_shapes
:
z
save/SaveV2/shape_and_slicesConst"/device:CPU:0*
valueBB B B B *
dtype0*
_output_shapes
:
×
save/SaveV2SaveV2save/ShardedFilenamesave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesglobal_step/Read/ReadVariableOp4linear/linear_model/bias_weights/Read/ReadVariableOp;linear/linear_model/curb-weight/weights/Read/ReadVariableOp;linear/linear_model/highway-mpg/weights/Read/ReadVariableOp"/device:CPU:0*
dtypes
2	
 
save/control_dependencyIdentitysave/ShardedFilename^save/SaveV2"/device:CPU:0*
T0*'
_class
loc:@save/ShardedFilename*
_output_shapes
: 
 
+save/MergeV2Checkpoints/checkpoint_prefixesPacksave/ShardedFilename^save/control_dependency"/device:CPU:0*
T0*
N*
_output_shapes
:
u
save/MergeV2CheckpointsMergeV2Checkpoints+save/MergeV2Checkpoints/checkpoint_prefixes
save/Const"/device:CPU:0

save/IdentityIdentity
save/Const^save/MergeV2Checkpoints^save/control_dependency"/device:CPU:0*
T0*
_output_shapes
: 
õ
save/RestoreV2/tensor_namesConst"/device:CPU:0*
valueBBglobal_stepB linear/linear_model/bias_weightsB'linear/linear_model/curb-weight/weightsB'linear/linear_model/highway-mpg/weights*
dtype0*
_output_shapes
:
}
save/RestoreV2/shape_and_slicesConst"/device:CPU:0*
valueBB B B B *
dtype0*
_output_shapes
:
®
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*
dtypes
2	*$
_output_shapes
::::
N
save/Identity_1Identitysave/RestoreV2*
T0	*
_output_shapes
:
T
save/AssignVariableOpAssignVariableOpglobal_stepsave/Identity_1*
dtype0	
P
save/Identity_2Identitysave/RestoreV2:1*
T0*
_output_shapes
:
k
save/AssignVariableOp_1AssignVariableOp linear/linear_model/bias_weightssave/Identity_2*
dtype0
P
save/Identity_3Identitysave/RestoreV2:2*
T0*
_output_shapes
:
r
save/AssignVariableOp_2AssignVariableOp'linear/linear_model/curb-weight/weightssave/Identity_3*
dtype0
P
save/Identity_4Identitysave/RestoreV2:3*
T0*
_output_shapes
:
r
save/AssignVariableOp_3AssignVariableOp'linear/linear_model/highway-mpg/weightssave/Identity_4*
dtype0

save/restore_shardNoOp^save/AssignVariableOp^save/AssignVariableOp_1^save/AssignVariableOp_2^save/AssignVariableOp_3
-
save/restore_allNoOp^save/restore_shardÄ=
ķ
_
.zero_fraction_total_zero_zero_count_true_10449
placeholder
placeholder_1		
constJ
ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: "
constConst:output:0*
_input_shapes
:: : :  
Ė
b
zero_fraction_cond_true_104597
3count_nonzero_notequal_zero_fraction_readvariableop
cast	X
count_nonzero/zerosConst*
valueB
 *    *
dtype0*
_output_shapes
: 
count_nonzero/NotEqualNotEqual3count_nonzero_notequal_zero_fraction_readvariableopcount_nonzero/zeros:output:0*
T0*
_output_shapes

:n
count_nonzero/CastCastcount_nonzero/NotEqual:z:0*

SrcT0
*

DstT0*
_output_shapes

:d
count_nonzero/ConstConst*
valueB"       *
dtype0*
_output_shapes
:y
count_nonzero/nonzero_countSumcount_nonzero/Cast:y:0count_nonzero/Const:output:0*
T0*
_output_shapes
: b
CastCast$count_nonzero/nonzero_count:output:0*

SrcT0*

DstT0	*
_output_shapes
: "
castCast:y:0*
_input_shapes

::  
²
z
zero_fraction_cond_false_104607
3count_nonzero_notequal_zero_fraction_readvariableop
count_nonzero_nonzero_count	X
count_nonzero/zerosConst*
valueB
 *    *
dtype0*
_output_shapes
: 
count_nonzero/NotEqualNotEqual3count_nonzero_notequal_zero_fraction_readvariableopcount_nonzero/zeros:output:0*
T0*
_output_shapes

:n
count_nonzero/CastCastcount_nonzero/NotEqual:z:0*

SrcT0
*

DstT0	*
_output_shapes

:d
count_nonzero/ConstConst*
valueB"       *
dtype0*
_output_shapes
:y
count_nonzero/nonzero_countSumcount_nonzero/Cast:y:0count_nonzero/Const:output:0*
T0	*
_output_shapes
: "C
count_nonzero_nonzero_count$count_nonzero/nonzero_count:output:0*
_input_shapes

::  
ļ
a
0zero_fraction_total_zero_zero_count_1_true_10492
placeholder
placeholder_1		
constJ
ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: "
constConst:output:0*
_input_shapes
:: : :  
²
z
zero_fraction_cond_false_105037
3count_nonzero_notequal_zero_fraction_readvariableop
count_nonzero_nonzero_count	X
count_nonzero/zerosConst*
valueB
 *    *
dtype0*
_output_shapes
: 
count_nonzero/NotEqualNotEqual3count_nonzero_notequal_zero_fraction_readvariableopcount_nonzero/zeros:output:0*
T0*
_output_shapes

:n
count_nonzero/CastCastcount_nonzero/NotEqual:z:0*

SrcT0
*

DstT0	*
_output_shapes

:d
count_nonzero/ConstConst*
valueB"       *
dtype0*
_output_shapes
:y
count_nonzero/nonzero_countSumcount_nonzero/Cast:y:0count_nonzero/Const:output:0*
T0	*
_output_shapes
: "C
count_nonzero_nonzero_count$count_nonzero/nonzero_count:output:0*
_input_shapes

::  
č
²
/zero_fraction_total_zero_zero_count_false_10450H
Dzero_fraction_readvariableop_linear_linear_model_curb_weight_weights)
%tofloat_zero_fraction_total_size_size	
mul”
zero_fraction/ReadVariableOpReadVariableOpDzero_fraction_readvariableop_linear_linear_model_curb_weight_weights*
dtype0*
_output_shapes

:T
zero_fraction/SizeConst*
value	B	 R*
dtype0	*
_output_shapes
: _
zero_fraction/LessEqual/yConst*
valueB	 R’’’’*
dtype0	*
_output_shapes
: 
zero_fraction/LessEqual	LessEqualzero_fraction/Size:output:0"zero_fraction/LessEqual/y:output:0*
T0	*
_output_shapes
: Ć
zero_fraction/condStatelessIfzero_fraction/LessEqual:z:0$zero_fraction/ReadVariableOp:value:0*1
else_branch"R 
zero_fraction_cond_false_10460*
output_shapes
: *
_lower_using_switch_merge(*
Tout
2	*0
then_branch!R
zero_fraction_cond_true_10459*
Tcond0
*
Tin
2*
_output_shapes
: e
zero_fraction/cond/IdentityIdentityzero_fraction/cond:output:0*
T0	*
_output_shapes
: 
$zero_fraction/counts_to_fraction/subSubzero_fraction/Size:output:0$zero_fraction/cond/Identity:output:0*
T0	*
_output_shapes
: 
%zero_fraction/counts_to_fraction/CastCast(zero_fraction/counts_to_fraction/sub:z:0*

SrcT0	*

DstT0*
_output_shapes
: |
'zero_fraction/counts_to_fraction/Cast_1Castzero_fraction/Size:output:0*

SrcT0	*

DstT0*
_output_shapes
: ¬
(zero_fraction/counts_to_fraction/truedivRealDiv)zero_fraction/counts_to_fraction/Cast:y:0+zero_fraction/counts_to_fraction/Cast_1:y:0*
T0*
_output_shapes
: q
zero_fraction/fractionIdentity,zero_fraction/counts_to_fraction/truediv:z:0*
T0*
_output_shapes
: f
ToFloatCast%tofloat_zero_fraction_total_size_size*

SrcT0	*

DstT0*
_output_shapes
: C
mul_0Mulzero_fraction/fraction:output:0ToFloat:y:0*
T0"
mul	mul_0:z:0*
_input_shapes
:: : :  
ī
¶
1zero_fraction_total_zero_zero_count_1_false_10493H
Dzero_fraction_readvariableop_linear_linear_model_highway_mpg_weights+
'tofloat_zero_fraction_total_size_size_1	
mul”
zero_fraction/ReadVariableOpReadVariableOpDzero_fraction_readvariableop_linear_linear_model_highway_mpg_weights*
dtype0*
_output_shapes

:T
zero_fraction/SizeConst*
value	B	 R*
dtype0	*
_output_shapes
: _
zero_fraction/LessEqual/yConst*
valueB	 R’’’’*
dtype0	*
_output_shapes
: 
zero_fraction/LessEqual	LessEqualzero_fraction/Size:output:0"zero_fraction/LessEqual/y:output:0*
T0	*
_output_shapes
: Ć
zero_fraction/condStatelessIfzero_fraction/LessEqual:z:0$zero_fraction/ReadVariableOp:value:0*1
else_branch"R 
zero_fraction_cond_false_10503*
output_shapes
: *
_lower_using_switch_merge(*
Tout
2	*0
then_branch!R
zero_fraction_cond_true_10502*
Tcond0
*
Tin
2*
_output_shapes
: e
zero_fraction/cond/IdentityIdentityzero_fraction/cond:output:0*
T0	*
_output_shapes
: 
$zero_fraction/counts_to_fraction/subSubzero_fraction/Size:output:0$zero_fraction/cond/Identity:output:0*
T0	*
_output_shapes
: 
%zero_fraction/counts_to_fraction/CastCast(zero_fraction/counts_to_fraction/sub:z:0*

SrcT0	*

DstT0*
_output_shapes
: |
'zero_fraction/counts_to_fraction/Cast_1Castzero_fraction/Size:output:0*

SrcT0	*

DstT0*
_output_shapes
: ¬
(zero_fraction/counts_to_fraction/truedivRealDiv)zero_fraction/counts_to_fraction/Cast:y:0+zero_fraction/counts_to_fraction/Cast_1:y:0*
T0*
_output_shapes
: q
zero_fraction/fractionIdentity,zero_fraction/counts_to_fraction/truediv:z:0*
T0*
_output_shapes
: h
ToFloatCast'tofloat_zero_fraction_total_size_size_1*

SrcT0	*

DstT0*
_output_shapes
: C
mul_0Mulzero_fraction/fraction:output:0ToFloat:y:0*
T0"
mul	mul_0:z:0*
_input_shapes
:: : :  
Ė
b
zero_fraction_cond_true_105027
3count_nonzero_notequal_zero_fraction_readvariableop
cast	X
count_nonzero/zerosConst*
valueB
 *    *
dtype0*
_output_shapes
: 
count_nonzero/NotEqualNotEqual3count_nonzero_notequal_zero_fraction_readvariableopcount_nonzero/zeros:output:0*
T0*
_output_shapes

:n
count_nonzero/CastCastcount_nonzero/NotEqual:z:0*

SrcT0
*

DstT0*
_output_shapes

:d
count_nonzero/ConstConst*
valueB"       *
dtype0*
_output_shapes
:y
count_nonzero/nonzero_countSumcount_nonzero/Cast:y:0count_nonzero/Const:output:0*
T0*
_output_shapes
: b
CastCast$count_nonzero/nonzero_count:output:0*

SrcT0*

DstT0	*
_output_shapes
: "
castCast:y:0*
_input_shapes

::  "w<
save/Const:0save/Identity:0save/restore_all (5 @F8"~
global_stepom
k
global_step:0global_step/Assign!global_step/Read/ReadVariableOp:0(2global_step/Initializer/zeros:0H"ü
	variablesīė
k
global_step:0global_step/Assign!global_step/Read/ReadVariableOp:0(2global_step/Initializer/zeros:0H
Ū
)linear/linear_model/curb-weight/weights:0.linear/linear_model/curb-weight/weights/Assign=linear/linear_model/curb-weight/weights/Read/ReadVariableOp:0(2;linear/linear_model/curb-weight/weights/Initializer/zeros:08
Ū
)linear/linear_model/highway-mpg/weights:0.linear/linear_model/highway-mpg/weights/Assign=linear/linear_model/highway-mpg/weights/Read/ReadVariableOp:0(2;linear/linear_model/highway-mpg/weights/Initializer/zeros:08
æ
"linear/linear_model/bias_weights:0'linear/linear_model/bias_weights/Assign6linear/linear_model/bias_weights/Read/ReadVariableOp:0(24linear/linear_model/bias_weights/Initializer/zeros:08"
trainable_variablesž
Ū
)linear/linear_model/curb-weight/weights:0.linear/linear_model/curb-weight/weights/Assign=linear/linear_model/curb-weight/weights/Read/ReadVariableOp:0(2;linear/linear_model/curb-weight/weights/Initializer/zeros:08
Ū
)linear/linear_model/highway-mpg/weights:0.linear/linear_model/highway-mpg/weights/Assign=linear/linear_model/highway-mpg/weights/Read/ReadVariableOp:0(2;linear/linear_model/highway-mpg/weights/Initializer/zeros:08
æ
"linear/linear_model/bias_weights:0'linear/linear_model/bias_weights/Assign6linear/linear_model/bias_weights/Read/ReadVariableOp:0(24linear/linear_model/bias_weights/Initializer/zeros:08"3
	summaries&
$
bias:0
fraction_of_zero_weights:0"%
saved_model_main_op


group_deps*ž
predictņ
/
curb-weight 
Placeholder:0’’’’’’’’’
1
highway-mpg"
Placeholder_1:0’’’’’’’’’p
predictionsa
Jlinear/linear_model/linear/linear_model/linear/linear_model/weighted_sum:0’’’’’’’’’tensorflow/serving/predict