# flashattn

*Author: Zihao Ye*

---

## From Online Softmax to FlashAttention

by Zihao Ye Email: zhye@cs.washington.edu May 11, 2023 UW CSE 599M Spring 2023: ML for ML Systems

The key innovation of FlashAttention [1] is using an idea similar to Online Softmax [3] to tile the self-attention computation, so that we can fuse the entire multi-head attention layer without accessing GPU global memory for intermediate logits and attention scores. In this note I'll briefly explain why tiling self-attention computation is non-trivial, and how to derive FlashAttention computation from online softmax trick. We thank Andrew Gu for proofreading this note.

### 1 The Self-Attention

The computation of Self-Attention can be summarized as (we ignore heads and batches because computation on these dimensions are fully parallel, we also omit details such as attention masks and scale factor 1 for simplicity): p D O = softmax ( QK T ) V (1)

where Q;K;V;O are both 2D matrix with shape ( L;D ) , where L is the sequence length and D is the dimension per head (a.k.a. head dimension), the softmax applies to the last dimension (columns). Thestandardapproachtocomputingself-attentionistofactorizethecomputationintoseveral stages:

X = QK T (2) A = softmax ( X ) (3) O = AV (4)

We call X matrix the pre-softmax logits, A matrix the attention score and O matrix the output. One amazingfactaboutFlashAttentionisthatwe don'tneedtomaterialize X and A matrices on global memory, instead we fuse the entire computation in formula 1 in a single CUDA kernel. This requires us to design an algorithm that carefully manages on-chip memory (like stream algorithms) because NVIDIA GPU's shared memory is small. For classical algorithms such as matrix multiplication, tiling is used to ensure that on-chip memorydoesnotexceedhardwarelimits. Figure1providesanexampleofthis. Duringkernelexe- cution, only 3 T 2 elements are stored on-chip, regardless of the matrix shape. This tiling approach is valid because addition is associative, allowing for the decomposition of the entire matrix multi- plication into the sum of many tile-wise matrix multiplications. However, Self-Attention includes a softmax operator that is not directly associative, making it hard to simply tile Self-Attention like in Figure 1. Is there a way to make softmax associative?

1

2 Section 2

|     |     |     |     |
| --- | --- | --- | --- |
|     |     |     |     |
|     |     |     |     |
|     |     |     |     |

|     |     |     |     |
| --- | --- | --- | --- |
|     |     |     |     |
|     |     |     |     |
|     |     |     |     |

|     |     |     |     |
| --- | --- | --- | --- |
|     |     |     |     |
|     |     |     |     |
|     |     |     |     |

Figure1. Thefigureaboutbrieflyexplainshowtotileinputandoutputmatricesformatrixmultiplication C = A B ,thematricesarepartitionedto T T tiles. Foreachoutputtile,wesweeptherelatedtilesin A forlefttoright,andrelatedtilesin B fortoptodown,andloadthevaluesfromglobalmemorytoon- chipmemory(coloredinblue,theoverallon-chipmemoryfootprintis O ( T 2 ) ). Fortile-wisepartialmatrix multiplication,forposition ( i;j ) ,weload A [ i;k ] and B [ k;j ] (coloredinred)forall k insidethetilefromon- chipmemory, thenaggregate A [ i;k ] B [ k;j ] to C [ i;j ] inon-chipmemory. Afterthecommputationofa tileiscomplete,wewritetheon-chip C tilebacktomainmemoryandmoveontotheprocessingthenexttile. Tilinginrealworldapplicationismuchmorecomplicated,youcanrefertoCutlassimplementationof matrixmultiplicationonA100[2].

### 2 (Safe) Softmax

Let's recall the softmax operator first, below is the generic formula of softmax computation:

e x i N softmax ( x 1 ;:::;x N )= (5) f g ( N e x j ) j =1 i =1 P Note that x i might be very large and e x i can easily overflow. For instance, the maximum number thatfloat16cansupportis65536,whichmeansthatfor x > 11, e x wouldexceedtheeffectiverange of float16. Tomitigatethisissue,mathematicalsoftwareoftenemploysatrickknownasthesafesoftmax: e x i e x i ¡ m = (6) N e x j N e x j m j =1 j =1 ¡ P P where m = max N j =1 ( x j ) , so that we can confirm each x i m 6 0 , which is safe because the expo- ¡ nential operator is accurate for negative inputs.

Online Softmax 3

Then we can summarize the computation of safe softmax as the following 3-pass algorithm:

Algorithm 3-pass safe softmax Notations m i : max i j =1 x j , with initial value m 0 = . f g f g ¡1 d i : i e x j ¡ m N , with initial value d 0 =0 , d N is the denominator of safe softmax. f g j =1 a i : the final softmax value. f g P Body for i 1 ;N do m i max ( m i 1 ;x i ) (7) ¡ end for i 1 ;N do d i d i 1 + e x i ¡ m N (8) ¡ end for i 1 ;N do e x i ¡ m N a i (9) d N end

This algorthm requires us to iterate over [1 ;N ] for 3 times. In the context of self-attention in Transformer,the x i arepre-softmaxlogitscomputedby QK T . Thismeansifwedon't alllogits x N (we don't f have g large enough SRAM to fit all of them), we need to access Q and K three f i g i =1 times (to re-compute logits on-the-fly), which is not I/O efficient.

### 3 Online Softmax

If we fuse the equation 7, 8 and 9 in a single loop, we can reduce the global memory access time from 3 to 1. Unfortunately, we cannot fuse equation 7 and 8 in the same loop because 8 depends on m N , which cannot be determined until the first loop completes. We can create another sequence d 0 := i e x j ¡ m i as a surrogate for original sequence d i := i j =1 i e x j ¡ m N to remove the dependency on N , and the N -th term of these two sequence are j =1 P identical: d N = d N 0 ,thuswecansafelyreplace d N inequation9with d N 0 .Wecanalsofindarecurrence P relation between d i 0 and d i 0 1 : ¡ i d i 0 = e x j ¡ m i j X =1 i 1 = ¡ e x j ¡ m i + e x i ¡ m i 0 1 j X =1 @ A i 1 = ¡ e x j ¡ m i 1 e m i 1 ¡ m i + e x i ¡ m i 0 ¡ 1 ¡ j X =1 = @ d 0 e m i 1 ¡ m i + A e x i ¡ m i (10) i 1 ¡ ¡ This recurrent form only relies on m i and m i 1 , and we can compute m j and d j 0 together in the same loop: ¡

4 Section 4

Algorithm 2-pass online softmax for i 1 ;N do m i max ( m i 1 ;x i ) ¡ d i 0 d i 0 1 e m i ¡ 1 ¡ m i + e x i ¡ m i ¡ end for i 1 ;N do e x i ¡ m N a i d N 0 end ThisisthealgorithmproposedinOnlineSoftmaxpaper[3]. However,itstillrequirestwopasses to complete the softmax calculation, can we reduce the number of passes to 1 to minimize global I/O?

### 4 FlashAttention

Unfortunately, the answer is no for softmax, but in Self-Attention, our final target is not the attentionscorematrix A ,butthe O matrixwhichequals A V . Canwefindaone-passrecurrence form for O instead? Let'strytoformulatethe k -throw(thecomputationofallrowsareindependent,andweexplain the computationofone rowfor simplicity)of Self-Attention computationas recurrence algorithm:

Algorithm Multi-pass Self-Attention Notations Q [ k; :] : the k -th row vector of Q matrix. K T [: ;i ] : the i -th column vector of K T matrix. O [ k; :] : the k -th row of output O matrix. V [ i; :]: the i -th row of V matrix. o i : i a j V [ j; :] , a row vector storing partial aggregation result A [ k; : i ] V [: i; :] f g j =1 P Body for i 1 ;N do x i Q [ k; :] K T [: ;i ] m i max ( m i 1 ;x i ) ¡ d i 0 d i 0 1 e m i ¡ 1 ¡ m i + e x i ¡ m i ¡ end for i 1 ;N do e x i ¡ m N a i (11) d N 0 o i o i 1 + a i V [ i; :] (12) ¡ end O [ k; :] o N

FlashAttention 5

Let's replace the a i in equation 12 by its definition in equation 11: i e x j m N o := ¡ V [ j; :] (13) i d j =1 N 0 X This still depends on m N and d N which cannot be determined until the previous loop completes. But we can play the surrogate trick in section 3 again, by creating a surrogate sequence o 0 : i e x j m i o 0 := ¡ V [ j; :] i 0 d 1 j =1 i 0 X @ A The n -th element of o and o 0 are the identical: o N 0 = o N , and we can find a recurrence relation between o i 0 and o i 0 1 : ¡ i e x j m i o 0 = ¡ V [ j; :] i d j =1 i 0 X i ¡ 1 e x j ¡ m i e x i m i = V [ j; :] + ¡ V [ i; :] 0 d i 0 1 d i 0 j X =1 @ A i ¡ 1 e x j ¡ m i ¡ 1 e x j ¡ m i d i 0 1 e x i ¡ m i = ¡ V [ j; :] + V [ i; :] 0 d i 0 1 e x j ¡ m i ¡ 1 d i 0 1 d i 0 j X =1 ¡ @ A i ¡ 1 e x j ¡ m i ¡ 1 d i 0 1 m m e x i ¡ m i = V [ j; :] ¡ e i ¡ 1 ¡ i + V [ i; :] 0 d i 0 1 1 d i 0 d i 0 j X =1 ¡ @ d i 0 1 e m i ¡ 1 ¡ m i e A x i ¡ m i = o i 0 1 ¡ + V [ i; :] (14) ¡ d i 0 d i 0 which only depends on d i 0 , d i 0 1 , m i , m i 1 and x i , thus we can fuse all computations in Self- Attention in a single loop: ¡ ¡ Algorithm FlashAttention for i 1 ;N do x i Q [ k; :] K T [: ;i ] m i max ( m i 1 ;x i ) ¡ d i 0 d i 0 1 e m i ¡ 1 ¡ m i + e x i ¡ m i ¡ d i 0 1 e m i ¡ 1 ¡ m i e x i ¡ m i o i 0 o i 0 1 ¡ + V [ i; :] ¡ d i 0 d i 0 end O [ k; :] o N 0

Thestates x i , m i , d i 0 ,and o i 0 havesmallfootprintsthatcaneasilyfitintoGPUsharedmemory. Becausealloperationsinthisalgorithmareassociative,itiscompatiblewithtiling. Ifwecompute the states tile-by-tile, the algorithm can be expressed as follows: Algorithm FlashAttention (Tiling) New Notations b : the block size of the tile # tiles: number of tiles in the row, N = b # tiles. x i : a vector storing the Q [ k ] K T value of the i -th tile [( i 1) b : ib ] . m ( local ) : the local maximum value inside x . ¡ i i

6 Bibliography

Body for i 1 ; # tiles do x i Q [ k; :] K T [: ; ( i 1) b : ib ] ¡ m ( local ) = max b ( x [ j ]) i i j =1 m max m ;m ( local ) i i 1 i ¡ b d d ¡ e m i 1 m i + e x i [ j ] m i i 0 i 0 1 ¡ ¡ ¡ ¡ j X =1 d 0 e m i 1 ¡ m i b e x i [ j ] ¡ m i o i 0 o i 0 1 i ¡ 1 ¡ + V [ j +( i 1) b; :] ¡ d i 0 d i 0 ¡ j X =1 end O [ k; :] o N 0 / b The figure 2 illustrate how to map this algorithm to hardware.

|     |
| --- |
|     |
|     |

|     |     |     |
| --- | --- | --- |

|     |
| --- |
|     |
|     |
|     |

|     |
| --- |
|     |
|     |

|     |
| --- |
|     |
|     |

|     |
| --- |
|     |
|     |

|     |
| --- |
|     |
|     |
|     |

Figure2. ThediagramaboveillustrateshowFlashAttentioniscomputedonhardware. Theblue-colored blocksrepresentthetilesthatresideinSRAM,whilethered-coloredblockscorrespondtothe i -throw. L denotesthesequencelength,whichcanbequitelarge(e.g.,16k), D denotestheheaddimension,whichis usuallysmallinTransformers(e.g.,128forGPT3),and B istheblocksizethatcanbecontrolled. Notably,theoverallSRAMmemoryfootprintdependsonlyon B and D andisnotrelatedto L . Asa result,thisalgorithmcanscaletolongcontextwithoutencounteringmemoryissues(GPUsharedmemory issmall,228kb/SMforH100architecture). Duringthecomputation,wesweepthetilesfromlefttoright for K T andA,fromtoptobottomfor V ,andupdatethestateof m , d ,and O accordingly.

### Bibliography

[1] Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, andChristopher Ré. Flashattention: fast andmemory- efficientexactattentionwithio-awareness. CoRR ,abs/2205.14135,2022. [2] AndrewKerr. Gtc 2020: developing cuda kernelsto pushtensor coresto the absolute limit onnvidia a100. May2020. [3] MaximMilakovandNataliaGimelshein. Onlinenormalizercalculationforsoftmax. CoRR ,abs/1805.02867, 2018.

