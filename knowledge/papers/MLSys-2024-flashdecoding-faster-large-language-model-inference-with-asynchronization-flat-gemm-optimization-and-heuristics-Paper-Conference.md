# MLSys-2024-flashdecoding-faster-large-language-model-inference-with-asynchronization-flat-gemm-optimization-and-heuristics-Paper-Conference

*Author: Ke Hong, Guohao Dai, Jiaming Xu, Qiuli Mao, Xiuhong Li, Jun Liu, Kangdi Chen, Yuhan Dong, Yu Wang*

---

#### F LASH D ECODING ++: F ASTER L ARGE L ANGUAGE M ODEL I NFERENCE WITH

### A SYNCHRONIZATION , F LAT GEMM O PTIMIZATION , AND H EURISTICS

KeHong *12 GuohaoDai *32 JiamingXu *32 QiuliMao 12 XiuhongLi 4 JunLiu 32 KangdiChen 2 YuhanDong 1 YuWang 1 A BSTRACT AstheLargeLanguageModel(LLM)becomesincreasinglyimportantinvariousdomains,theperformanceof LLMinferenceiscrucialtomassiveLLMapplications. However,thefollowingchallengesstillremainunsolved in accelerating LLM inference: (1) Synchronized partial softmax update. The softmax operation requires a synchronizedupdateoperationamongeachpartialsoftmaxresult,leadingto ∼ 20%overheadsfortheattention computationinLLMs. (2)Under-utilizedcomputationofflatGEMM.TheshapeofmatricesperformingGEMM inLLMinferenceisflat,leadingtounder-utilizedcomputationand50%performancelossafterpaddingzerosin previousdesigns( e.g., cuBLAS,CUTLASS,etc.).(3)Performancelossduetostaticdataflow.Kernelperformance inLLMdependsonvariedinputdatafeatures,hardwareconfigurations,etc. Asingleandstaticdataflowmaylead toa50.25%performancelossforGEMMsofdifferentshapesinLLMinference. Wepresent FlashDecoding++ ,afastLLMinferenceenginesupportingmainstreamLLMsandhardwareback-ends. Totackletheabovechallenges, FlashDecoding++ creativelyproposes: (1)Asynchronizedsoftmaxwithunified maxvalue. FlashDecoding++ introducesaunifiedmaxvaluetechniquefordifferentpartialsoftmaxcomputations toavoidsynchronization.Basedonthis,thefine-grainedpipeliningisproposed,leadingto1.18 × and1.14 × forthe prefillanddecodingstageinLLMinference,respectively. (2)FlatGEMMoptimizationwithdoublebuffering. FlashDecoding++ pointsoutthatflatGEMMswithdifferentshapesfacevariedbottlenecks. Then,techniques likedoublebufferingareintroduced,resultinginupto52%speedupfortheflatGEMMoperation. (3)Heuristic dataflowwithhardwareresourceadaptation. FlashDecoding++ heuristicallyoptimizesdataflowusingdifferent hardwareresource( e.g., Tensor CoreorCUDAcore)consideringinputdynamics. The designleadstoupto 29%speedupcomparedwiththestaticdataflow. Duetotheversatilityofoptimizationsin FlashDecoding++ , FlashDecoding++ canachieveupto 4.86 × and 4.35 × speeduponbothNVIDIAandAMDGPUscompared toHuggingFaceimplementations. FlashDecoding++ alsoachievesanaveragespeedupof 1.37 × comparedto state-of-the-artLLMinferenceenginesonmainstreamLLMs.

| Token/s input length = 1K
30
NVIDIA Tesla A100 F la s h D e c o d i n g + + ( o u r s ) faster
25 AMD MI210 20 tuphguorht nekot sm/ycnetal H u g g in g F F l a a s c h e D /P e y c T o o d r i n c g h DeepSpeed
15 hcae OpenPPL×
vllm+
10
107 5 first token
70 90 110 13 92 ecnerefni latency/ms
83
input length = 32K
80
70 MLL faster
38 60 nekot sm/ycnetal
50 hcae
40
30 SOTA w/ FlashDecoding++ 3200 3800 4400 500 first token latency/ms |                                | input length = 1K |     |     |     |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------ | ----------------- | --- | --- | --- |
|                                                                                                                                                                                                                                                                                                                                                                                                                                                   | 25
nekot sm/ycnetal
20
15 hcae | as                |     | ter |     |
|                                                                                                                                                                                                                                                                                                                                                                                                                                                   |                                | f                 |     |     |     |
|                                                                                                                                                                                                                                                                                                                                                                                                                                                   |                                |                   |     |     |     |
|                                                                                                                                                                                                                                                                                                                                                                                                                                                   |                                |                   |     |     |     |

#### 1 I NTRODUCTION

As the Large Language Model (LLM) achieved unprece- dentedsuccessinvariousdomains(Thirunavukarasuetal., 2023;Aniletal.,2023;Clusmannetal.,2023;Cuietal., 2023), theLLMinferenceworkloadisskyrocketing. For 130

|                                   | input length = 32K |      |     |
| --------------------------------- | ------------------ | ---- | --- |
|                                   |                    |      |     |
| 70
nekot sm/ycnetal
60
50 hcae
40 |                    |      |     |
|                                   |                    | ster |     |
|                                   | f                  | a    |     |
|                                   |                    |      |     |

example,OpenAIreportsthatGPT-4inferencewith8Kcon- textlengthcosts$0.03per1Kinputtokensand$0.06per 1Koutputtokens(OpenAI,2023). Currently,OpenAIhas 180.5 million users and receives over 10 million queries per day (Nerdynav, 2023). Consequently, the cost to op- erateOpenAI’smodellikeChatGPTisapproximately$7 5000 Figure1. Overviewofcomparisonbetween FlashDecoding++ and * Equal contribution 1 Tsinghua University 2 Infinigence-AI state-of-the-artdesigns.Theresultsinthefigurearereportedwith 3 ShanghaiJiaoTongUniversity 4 PekingUniversity. Correspon- Llama2-7Bmodel(Touvronetal.,2023). Theleftiswithbatch dence to: Guohao Dai < daiguohao@sjtu.edu.cn > , Yu Wang < yuwang@tsinghua.edu.cn > . size=1andinputlength=1K,andTensorRT-LLMandHugging FacearetheSOTAbaselineforNVIDIA/AMDaccordingtoour Proceedingsofthe 7 th MLSysConference ,SantaClara,CA,USA, experimentalresults.Therightshowsthecomprehensivecompari- 2024.Copyright2024bytheauthor(s). sonofbothfirsttokenlatencyandeachtokenlatency.

FlashDecoding++:FasterLargeLanguageModelInferencewithAsynchronization,FlatGEMMOptimization,andHeuristics (a) partial attention Synchronized partial softmax update (b)

|     |     |     |     |
| --- | --- | --- | --- |
|     |     |     |     |
|     |     | K   |     |
| Q
V |     |     |     |
|     |     |     |     |
|     |     |     |     |
|     |     | Q   |     |
|     |     |     |     |
|     |     |     |     |
|     |     | V   |     |
|     |     |     |     |
|     |     |     |     |

① partial softmax × W (e.g., FlashAttention) Attention

| mul1 | max | exp |
| ---- | --- | --- |

| ul2 | Attention
N+1 |
| --- | ------------- |

N-1 sum mul

|     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- |
|     |     |     |     |     |     |
|     |     |     |     | P   |     |

|     |     |     |     |
| --- | --- | --- | --- |
|     |     |     |     |
| tt  | ent | io  | n   |

e s synchronized update mul 1 & mul 2 refer to ahp × operation ② & ④ in (a)

| W    | hat     |
| ---- | ------- |
|      | is      |
| larg | the
est |

|     |     |     |
| --- | --- | --- |
|     |     |     |
|     |     |     |

|     |     |     |
| --- | --- | --- |
|     |     |     |
|     |     |     |

|     |     |     |     |
| --- | --- | --- | --- |
|     |     |     |     |
|     |     |     |     |

|     |     |
| --- | --- |
|     |     |
|     |     |

② P Attention ④ Section 3 × W × lli ocean ③ softmax Attention N-1 mul 1 exp mul 2 Attention N+1 f e ? Pacific r P × W O FFN 1 FFN 2 unified max value sum asynchronized × W ⑤ ⑥ ⑥ Asynchronized softmax with unified max value Under-utilized computation of flat GEMM (c) padding zeros directly computing no ① ② ③ ④ ⑤ ⑥ A

| GEMM               |
| ------------------ |
| Q, K, V projection |
| GEMV/Flat GEMM     |

| GEMM           |
| -------------- |
| Q × K          |
| GEMV/Flat GEMM |

| softmax |
| ------- |
| softmax |
| softmax |

| GEMM           |
| -------------- |
| Attention × V  |
| GEMV/Flat GEMM |

| GEMM           |
| -------------- |
| O projection   |
| GEMV/Flat GEMM |

| GEMM           |
| -------------- |
| Feedforward    |
| GEMV/Flat GEMM |

| load A | A×B | load A’ |
| ------ | --- | ------- |

i t flat-shape zero × B or a r GEMM computation under-utilization ep A × B Section 4 O

|     |     |     |
| --- | --- | --- |
|     |     |     |
| K   | c   | ac  |
|     |     |     |

| A×B     | load A’’ | A’’×B     |        |
| ------- | -------- | --------- | ------ |
| load A’ | A’×B     | load A’’’ | A’’’×B |

① partial attention load A × W K (e.g., FlashDecoding) double buffering e cache Flat GEMM optimization with double buffering s ahp × Performance loss to static dataflow (d) W ② ③ softmax ④

| GEMM           |
| -------------- |
| Flat GEMM
GEMV |

| static dataflow 1 |
| ----------------- |
| static dataflow 2 |

edo Pacific × Q × Ocean GEMM√ Flat GEMM × GEMV × × W O FFN 1 FFN 2 c ⑤ ⑥ ⑥ GEMM × Flat GEMM√ GEMV√

|     |     |     |
| --- | --- | --- |
|     |     |     |
|     |     |     |
| V   | c   | ac  |
|     |     |     |

e D × W V Section 5

| GEMM      |
| --------- |
| Flat GEMM |
| GEMV      |

cache heuristic autogressively dataflow GEMM√ Flat GEMM√ GEMV√ Heuristic dataflow with hardware resource adaption Figure2. OverviewofLargeLanguageModelinferencedataflow. FlashDecoding++ proposesthreesolutionsforcorrespondingchallenges inLargeLanguageModelinference.(a)Thedataflowcomparisonbetweenthe prefill phaseandthe decode phase.The prefill phasemainly involvestheGEMMoperation,whilethe decode phasemainlyinvolvestheGEMV/FlatGEMMoperation.(b) FlashDecoding++ proposes theasynchronizedsoftmaxwithunifiedmaxvaluetechnique,avoidingsynchronizedupdatetopreviouspartialattentionresults. (c) FlashDecoding++ optimizesflatGEMMbyimprovingcomputationutilization.(d) FlashDecoding++ heuristicallyoptimizesdataflow. millionperdayforthenecessarycomputinghardware(DY- Figure2(a)showsthemaindataflowoftheLLMinference LANPATEL,2023).Thus,optimizationsonLLMinference with one transformer layer for both the prefill phase and performancewillhaveahugeimpactconsideringmassive the decode phase. Atransformerlayercanbedividedinto LLM inference scenarios. Many recent works have pro- linearGEMM(GeneralMatrixMultiplication)operations posedtechniquestoaccelerateLLMinferencetasks,includ- ( e.g., K, Q, V, O weight projection and the feedforward) ingDeepSpeed(Aminabadietal.,2022),FlexGen(Sheng and the attention/softmax computation. For the attention etal.,2023),vLLM(Kwonetal.,2023),OpenPPL(Sense- computation, asoftmaxoperationisadoptedforarowin time,2023a),FlashDecoding(Daoetal.,2023),TensorRT- theattentionmatrix. Toimprovetheparallelism,previous LLM(Vaidyaetal.,2023),andetc(Sensetime,2023b;TGI, designs (Dao et al., 2022; 2023) divide the attention ma- 2023;mlc,2023;Sensetime,2023a). tricesintosmallertilesandrowsarealsosplittocompute partialsoftmaxresults. Asynchronizedsoftmaxoperation TheLLMinferencetaskgeneratestokens( e.g., words)from isadoptedtoupdatepreviouspartialsoftmaxresultswhen theinputsequenceautoregressively,andcanbeorganized anewpartialsoftmaxresultiscalculated. Sucha synchro- into two typical phases: the prefill phase and the decode nizedpartialsoftmaxupdate accountsfor18.8%forthe phase. The prefill phasegeneratesthefirsttokenbyprocess- attentioncomputation ofLlama2-7Binferenceaccording ingtheinputprompt,andpreviousresearch( e.g., FlashAt- to our profiling on NVIDIA Tesla A100 GPU with 1024 tention(Daoetal.,2022;Dao,2023))optimizeslatencyfor input length, resulting in the first challenge for accelerat- this phase. The decode phase generates the following to- ingLLMinference. Secondly, thecomputationresources kenssequentially,andmanyworks(Aminabadietal.,2022; is under-utilized for the flat GEMM operation during Shengetal.,2023;Kwonetal.,2023;Sensetime,2023b; the decode phase. Becausethe decode phasesequentially Daoetal.,2023;Vaidyaetal.,2023;Phametal.,2023)fo- generatestokens,thelinearGEMMoperationtendstobe cusonimprovingthethroughputofgeneratingtokens( i.e., flat-shape(eventurningintotheGEMV(GeneralMatrix- reducinglatencyofeachtoken).The prefill phasedominates VectorMultiplication)operationwhenthebatchsizeis1). totaltimeforscenariosoflong-sequenceinputorgenerat- Forthesmallbatchsize( e.g., 8),previousdesigns(NVIDIA, ingshortoutputs(Daietal.,2019;Dongetal.),whilethe 2017c;a)padthematrixwithzerostoperformGEMMsof decode phaseconstitutesasignificantportionofthetime larger sizes ( e.g., 64), leading to over 50% computation whenprocessinglongoutputsequences(Xiaoetal.,2023).

FlashDecoding++:FasterLargeLanguageModelInferencewithAsynchronization,FlatGEMMOptimization,andHeuristics under-utilization. Thirdly, the performance of LLM in- ferencesuffersfromthestaticdataflow consideringinput dynamics and hardware configuration. For example, the smallbatchsizemakesthe decode phaseofLLMinference memory-boundedandthelargebatchsizemakesitcompute- bounded. Asingleandstaticdataflowmayleadto50.25% performancelossforGEMMsofdifferentshapesinLLM inference. TotacklethesechallengesandenableafasterLargeLan- guage Model (LLM) inference, we present FlashDecod- ing++ inthispaper. FlashDecoding++ creativelyproposes thefollowingcontributions:

• Asynchronized softmax with unified max value. FlashDecoding++ leveragesaunifiedmaxvaluefor different partial softmax computations. Each partial softmaxresultcanbeprocessedindividuallywithout synchronizedupdate. Suchatechniqueleadsto1.18 × and 1.14 × speedup for attention computation in the prefill stageand decoding stage,respectively. • Flat GEMM optimization with double buffering. FlashDecoding++ onlypadsthematrixsizeto8rather than64inpreviousdesignsforflat-shapedGEMMto improvecomputationutilization. Wepointoutthatflat GEMMswithdifferentshapesfacevariedbottlenecks, andfurtherimprovethekernelperformancebyupto 52%withtechniqueslikedoublebuffering. • Heuristic dataflow with hardware resource adap- tion. FlashDecoding++ takes both input dynamics and hardware configurations into consideration and dynamicallyapplieskerneloptimizationfortheLLM inference dataflow. Such a technique leads to up to 29%speedup.

Because of the versatility of optimizations, the effective- nessof FlashDecoding++ canbeprovedonbothNVIDIA andAMDGPUs. FlashDecoding++ achievesupto 4.86 × and 4.35 × speeduponbothNVIDIAandAMDGPUscom- pared with Hugging Face implementations, respectively. Ourextensiveresultsshowthat FlashDecoding++ achieves anaverageof 1.37 × speedupcomparedwithFlashDecod- ing (Dao et al., 2023), a state-of-the-art LLM inference engineonvariousLLMs( e.g., Llama2,ChatGLM2,etc.). Therestofthispaperisorganizedasfollows. Section2in- troducespreliminariesofLLMsandrelatedworksonLLM inferenceacceleration. Ourthreetechniques,theasynchro- nizedsoftmaxwithunifiedmaxvalue,theflatGEMMopti- mizationwithdoublebuffering,andtheheuristicdataflow withhardwareresourceadaptionaredetailedinSection3,4, and 5, respectively. Section 6 presents the evaluation re- sults. RelatedworksonLLMinferenceareintroducedin Section7,andSection8concludesthepaper.

#### 2 B ACKGROUND

2.1 LLMInferenceDataflowOverview ThetaskofLLMinferenceistogeneratetokensfromthe inputsequence,whichcanbeusedtocompleteasentence oransweraquestion. AnoverviewoftheLLMinference dataflowisshowninFigure2(a). Aswecansee,theLLM inferencedataflowcanbeorganizedintotwotypicalphases withsimilaroperations:one prefill phaseandseveral decode phases. The prefill phase“understands”theinputsequence ( i.e., “Whatisthelargestocean?”). Eachtoken(wesetone wordasatokeninFigure2(a))isencodedasanembedding vector, and the input sequence is organized into a matrix. Themainoutputofthe prefill phaseisanewtoken,whichis predictedtobethenexttokenaftertheinputsequence( i.e., “Pacific”inthisfigure). The decode phase“generates”the outputsequence( i.e., “Pacific”,“Ocean”,etc.) Theoutput tokenofthe prefill phaseistakenastheinputofthe decode phase. The decode phase is executed autogressively, and eachoutputtokenisusedastheinputtokenforthenextThe decode ( e.g., “Ocean”isfurtherusedastheinput).

2.2 OperationsinLLMInference ThemainoperationsinLLMinferencearedepictedasoper- ation ① to ⑥ inFigure2(a),includingthelinearprojection ( ① and ⑤ ),theattention( ② , ③ ,and ④ ),andthefeedforward network( ⑥ ). Forsimplicity,operationslikepositionembed- ding(Vaswanietal.,2017),non-linearactivation(Nair& Hinton,2010;Ramachandranetal.,2017),mask(Vaswani etal.,2017),andothersarenotshowninthefigure. Opera- tionsinthe prefill phaseandthe decode phasearedifferent intheshapeofdata. Becauseonlyonetoken(batchsize = 1) orfewtokens(batchsize > 1)areprocessedatonetime, in- putmatricesinthe decode phaseareflat-shapematrices orevenvectors. LinearProjection. Thelinearprojectionperformsasthe fullyconnectedlayer,multiplyingtheinputwithweightma- trices( i.e., W K ,W Q ,W V ,W O ,called K,Q,V projection and O projection). Forthe prefill phase,the K,Q,V projec- tiongeneratesmatrices K,Q,V . Forthe decode phase,the K,Q,V projectiongeneratesthreecorrespondingvectors andconcatenatedwith K and V ( i.e., KVcache,yellowand lightblueinFigure2(a))inthe prefill phase. softmax ( Q × K T ) × V (1) Attention. Theattentionoperationismainlydividedinto threeoperations( ② to ④ Q × K , softmax , Attention × V ), asshowninEq.(1). For P = Q × K T , thesoftmax operationisperformedforeachrowoftheresultmatrixof P . ThedetailedsoftmaxcomputationisshowninFigure3(a). Themaximumvalue m ( x ) isfirstcalculated. Theexponent ofeachelementdividedby e m ( x ) , f ( x ) ,isthenprocessed. These exponents are normalized to the summation of all

FlashDecoding++:FasterLargeLanguageModelInferencewithAsynchronization,FlatGEMMOptimization,andHeuristics 𝑠𝑜𝑓𝑡𝑚𝑎𝑥 𝑥 𝑠𝑜𝑓𝑡𝑚𝑎𝑥 𝑥′ 𝑠𝑜𝑓𝑡𝑚𝑎𝑥 𝑥′′ 𝑠𝑜𝑓𝑡𝑚𝑎𝑥 𝑥′ 𝑠𝑜𝑓𝑡𝑚𝑎𝑥 𝑥′′

|     |     |     |     |     |       |     |
| --- | --- | --- | --- | --- | ----- | --- |
| x
1 | x
2 | x
3 | x
4 | …   | x
d-1 | x
d |
|     |     |     |     |     |       |     |

|     |     |     |     |     |       |     |
| --- | --- | --- | --- | --- | ----- | --- |
| x
1 | x
2 | x
3 | x
4 | …   | x
d-1 | x
d |
|     |     |     |     |     |       |     |

|     |     |     |     |     |       |     |
| --- | --- | --- | --- | --- | ----- | --- |
| x
1 | x
2 | x
3 | x
4 | …   | x
d-1 | x
d |
|     |     |     |     |     |       |     |

𝑚𝑥 =𝑚𝑎𝑥 𝑥 ! Calcutate 𝑚𝑥′ , 𝑓𝑥′ , 𝑙 𝑥′ , 𝑠𝑜𝑓𝑡𝑚𝑎𝑥 𝑥′ 𝑚𝑥′ =𝑚𝑥′′ = aunifiedmaxvalue 𝜙 𝑓𝑥 = 𝑒 " ' #$" ,𝑒 " ( #$" ,…,𝑒 " ) #$" Calcutate 𝑚𝑥′′ , 𝑓𝑥′′ , 𝑙 𝑥′′ , 𝑠𝑜𝑓𝑡𝑚𝑎𝑥 𝑥′′ Calcutate 𝑓𝑥′ , 𝑙 𝑥′ Update 𝑠𝑜𝑓𝑡𝑚𝑎𝑥 𝑥′ with 𝑚𝑥′ , 𝑓𝑥′ , 𝑙 𝑥′ , 𝑚𝑥′′ , 𝑓𝑥′′ , 𝑙 𝑥′′ Calcutate 𝑓𝑥′′ , 𝑙 𝑥 % ′ 𝑙 𝑥 =+ 𝑓 ! 𝑥 ! Update 𝑠𝑜𝑓𝑡𝑚𝑎𝑥 𝑥′′ with 𝑚𝑥′ , 𝑓𝑥′ , 𝑙 𝑥′ , 𝑚𝑥′′ , 𝑓𝑥′′ , 𝑙 𝑥′′ …… 𝑓𝑥 𝑠𝑜𝑓𝑡𝑚𝑎𝑥 𝑥 = 𝑙 𝑥 Processnextpartialvector Calcutate 𝑠𝑜𝑓𝑡𝑚𝑎𝑥 𝑥 Highparallelism × Highparallelism√ Highparallelism√ Lowmemory × Lowmemory√ Lowmemory√ Synchronization-free√ Synchronization-free × Synchronization-free√ (a)Originalsoftmax (b)Partialsoftmax (c)Partialsoftmaxwithunifiedmaxvalue Figure3. Comparisonofdifferentsoftmaxcomputationschemes.(a)Softmaxcomputationforthewholevector.(b)Computingpartial softmaxforeachpartialvector,andasynchronizedupdateoperationisrequiredforallpartialsoftmaxresults. (c)Computingpartial softmaxusingaunifiedmaxvalue,andeachpartialvectorisprocessedindividuallywithoutsynchronizedupdate. exponents( i.e., l ( x ) )togetthesoftmaxresult. 3 A SYNCHRONIZED S OFTMAX WITH U NIFIED M AXIMUM V ALUE FeedforwardNetwork. Thefeedforwardnetworkprimar- ilycomprisestwofullyconnectedlayers. Thefirstone( ⑥ Motivation. Thepartialsoftmaxoperationrequiressynchro- FFN 1 )expandsthefeaturedimensionstoenhancetherep- nizationamongdifferentpartialvectors,leadingto ∼ 20% resentationalcapacity. Thesecondone( ⑥ FFN 2 )restores overheadsoftheattentionoperation. AsisshowninFig- thefeaturedimensionsandservesastheoutputlayer. ure2(b),thesynchronizationisrequiredafterthemaximum valueofthepartialvectoriscalculated.Themaximumvalue 2.3 AttentionOptimization isusedtoupdatepreviouspartialsoftmax( i.e., recompute previous attention) results. Thus, to reduce synchroniza- The softmax operation shown in Figure 3(a) requires all tion overheads, the key problem to be solved is how to global data to be calculated and stored before it can pro- computeeachpartialsoftmaxresultwithoutrequiring ceed. This results in high memory consumption and low resultsfromotherpartialsoftmaxcomputation. parallelism. Latterworksproposethepartialsoftmaxtech- nique to reduce memory consumption (Dao et al., 2022; Challenge. The reason that synchronization is required Dao,2023)orimproveparallelism(Daoetal.,2023). Fig- lies in that the maximum value of each partial vector is ure3(b)showsthediagramofthepartialsoftmaxoperation. different. Themaximumvalueisusedtoavoidoverflowof Themainideaistodividethevector x intopartialvectors theexponentoperation( f ( x ) inFigure3(a)),andexponents ( i.e, x ′ and x ′′ ). The partial softmax results of x ′ and x ′′ aresummed( l ( x ) inFigure3(a))asthedenominatorofthe arecalculatedseparatelyaccordingtoFigure3(a),andthen softmax operation. Such a non-linear operation on each synchronouslyupdatedbyeachother. Thedetailedcompu- partialmaximumvaluemakesthesynchronizationamong tationofthissynchronizedupdateisshowninEquation(2). eachpartialsoftmaxcomputationunavoidable. Withtheimplementationofpartialsoftmax,wecanachieve AnalysisandInsights. Accordingtotheformulaofsoftmax efficientparallelismofcomputationwhilereducingmemory computation,themaximumvalueisusedasthescalingfac- costforattentioncomputation. torforboththenumeratorandthedenominator( i.e., f ( x ) m ( x )= max ( m ( x ′ ) ,m ( x ′′ )) and l ( x ) in Figure 3(a)). Our key insight is, the scaling f ( x ′ )= e m ( x ′ ) − m ( x ) f ( x ′ ) factorcanbeanarbitrarynumber ratherthanusingthe maximum value mathematically, shown in Equation (3). f ( x ′′ )= e m ( x ′′ ) − m ( x ) f ( x ′′ ) (2) Whenweset ϕ =0 ,itbecomestheoriginalsoftmaxcom- l ( x )= f ( x ′ )+ f ( x ′′ ) putation(Bridle,1989). softmax ([ x ′ ,x ′′ ])=[ f ( x ′ ) ,f ( x ′′ )] ÷ l ( x ) [ e x 1 − m ( x ) ,...,e x d − m ( x ) ] However, since the partial softmax needs to be updated softmax ( x )= (cid:80) e x i − m ( x ) according to other partial softmax results, it unavoidably i (3) [ e x 1 − ϕ ,...,e x d − ϕ ] introducesdatasynchronizationoperations. Accordingto = , ∀ ϕ ∈ R ourprofilingresult,suchasynchronizedupdateoperation (cid:80) e x i − ϕ i leadsto18.8%overheadsintheattentioncomputationfor Llama2-7B inference on NVIDIA Tesla A100 GPU with However,thescalingfactorcannotbeanarbitrarynumber 1024inputlength. consideringtheoverflowingoftheexponentcomputation.

FlashDecoding++:FasterLargeLanguageModelInferencewithAsynchronization,FlatGEMMOptimization,andHeuristics

|     |     |     | Llama2-7B OPT-6.7B ChatGLM2-6B |       |     |     |      |     |       |     |      |      |      |      |     |     |       |      |     |     |     |     |      |     |     |     |
| --- | --- | --- | ------------------------------ | ----- | --- | --- | ---- | --- | ----- | --- | ---- | ---- | ---- | ---- | --- | --- | ----- | ---- | --- | --- | --- | --- | ---- | --- | --- | --- |
|     |     |     |                                |       |     |     |      |     |       |     |      |      |      |      |     |     |       |      |     |     |     |     |      |     |     |     |
|     |     |     |                                |       |     |     |      |     |       |     |      |      |      |      |     |     |       |      |     | 99  |     |     | %    |     |     |     |
|     |     |     |                                |       | 99. | 99% |      |     |       |     |      |      | 9    | 9.99 |     | %   |       |      |     |     | 99  | .99 | %    |     |     |     |
|     |     |     |                                |       |     |     |      |     |       |     |      |      |      |      |     |     |       |      |     |     |     |     |      |     |     |     |
|     |     |     |                                |       |     |     |      |     |       |     |      |      |      |      |     |     |       |      |     |     |     |     |      |     |     |     |
|     |     |     |                                |       |     |     |      |     |       |     |      |      |      |      |     |     |       |      |     |     |     |     |      |     |     |     |
|     |     |     |                                |       |     |     |      |     |       |     |      |      |      |      |     |     |       |      |     |     |     |     |      |     |     |     |
|     |     |     |                                |       |     |     |      |     |       |     |      |      |      |      |     |     |       |      |     |     |     |     |      |     |     |     |
|     |     |     | [-                             | 16.8] |     |     | [6.5 | ]   |       |     | [-4  | 96.8 |      | ] [  |     | 363 | .5]   | [-10 |     | .5] |     |     | [13. |     | 7]  |     |
|     |     |     |                                |       |     |     |      |     |       |     |      |      |      |      |     |     |       |      |     |     |     |     |      |     |     |     |
|     |     |     | 70                             | -20   | -10 | 0   | 10   | 40  | -1400 |     | -400 |      | -200 |      |     | 0 2 | 00400 | -10  | -5  | 0   |     | 5   |      | 1   | 0 1 |     |

x v

| x=4
1 | x=5
2 | x=6
3 | x=7
4 |
| ----- | ----- | ----- | ----- |

| v
1 | v
2 | v
3 | v
4 |
| --- | --- | --- | --- |

get x 1 ,x 2 get x 3 ,x 4 fromQ,K fromQ,K calculate numerator+= calculate numerator+= numerator ! e 4-6 ,e 5-6 e 4-6 ·v 1 + e 5-6 ·v 2 e 6-6 ,e 7-6 e 4-6 ·v 1 + e 5-6 ·v 2 denominator denominator+= denominator+= e 4-6 + e 5-6 e 4-6 + e 5-6 (a)Calculate softmax(x) × v T y v

| y=3
1 | y=6
2 | y=10
3 | y=7
4 |
| ----- | ----- | ------ | ----- |

v 1 v 2 v 3 v 4 get y 1 ,y 2 get y 3 ,y 4 recomputationprocess fromQ,K fromQ,K Figure4. Thestatisticaldistributionof x i (elementsintheinput calculate numerator+= calculate compute compute update update vectorsofsoftmax)intypicalLLMswithdifferentinputs. e 3-6 ,e 6-6 e 3-6 ·v 1 + e 6-6 ·v 2 e 10-6 ,e 7-6 softmax 1 softmax 2 softmax 1 softmax 2 denominator+= 10-6>3,overflow! Forthecasewhere x ≫ ϕ , e x i − ϕ overflowsandcannotbe e 3-6 + e 6-6 i (b)Calculate softmax(y) × v T represented using a fix-width floating point number ( e.g., float32 for exponent results in current LLM engines). Figure5. Exampleofasynchronizedpartialsoftmaxcomputation. For another case where x ≪ ϕ , e x i − ϕ → 0 , leading to (a)Eachpartialsoftmaxresultisprocessindividuallywithoutthe i synchronizedupdate.(b)Therecomputationprocessforallparital precision loss. Thus, a proper scaling factor ϕ should be softmaxcomputationisrequiredwhenoverflowhappens. carefullyselectedtoavoidthetwocasesabove. Figure4 showsthestatisticaldistributionof x (elementsinthein- x i − ϕ ≥ b ,theasynchronizedpartialsoftmaxcomputation i putvectorsofsoftmax)intypicalLLMswithdifferentin- is terminated for the vector x where x i belongs to. The puts(Merityetal.,2016). Ourkeyinsightis, > 99 . 99% x softmaxisthenrecomputedusingthesynchronizedpartial i arewithinacertainrange . Specifically,forLlama2-7B, softmaxscheme(usedinFlashAttention(Daoetal.,2022; we have − 16 . 8 < x i < 6 . 5 for > 99 . 99% x i . Because Dao,2023)andFlashDecoding(Daoetal.,2023))shown e b − a and e a − b canberepresentedbya float32 format, inFigure3(b). Sucharecomputationschemeavoidsover- we can set ϕ = a in Equation (3). For OPT-6.7B, we do flowwhileintroducingnegligibleoverheadsbasedonthe notapplythetechniqueinthissectionbecauseofthelarge statisticaldatashowninFigure4. rangeinFigure4. Example. Figure5showsanexampleoftheasynchronized Approach:Asynchronization. Basedontheinsightsabove, softmax scheme. We set a = − 3 ,b = 3 ,ϕ = 6 . Two each partial softmax computation shares a unified maxi- vectors x and y arecalculatedfrom Q × K T inEquation(1), mumvalue, ϕ . Afterthesoftmaxoperation,aninnerprod- andaredividedinto2partialvectors. Weomittheprocess uct operation is executed between the softmax result and from Q × K T to these partial vectors. For each x i , we a column of V ( i.e., v ). Assume that the input vector x have a < x i − ϕ < b ,weprocess e x 1 − ϕ · v 1 + e x 2 − ϕ · v 2 can be divided into p partial vectors, x = [ x (1) ,...,x ( p ) ] and e x 1 − ϕ + e x 2 − ϕ for the first partial vector of x using ( v =[ v (1) ,...,v ( p ) ] correspondingly),wehave: twoasynchronizedthreads. Then,eachthreadmovestothe nextpartialvectorforthecorrespondingcomputation( i.e., (cid:80) e x 3 − ϕ · v + e x 4 − ϕ · v and e x 3 − ϕ + e x 4 − ϕ ). Twothreads e x i − ϕ · v 3 4 ⟨ softmax ( x ) ,v ⟩ = (cid:80) i i aresynchronizedwhenallpartialvectorsareprocessed,and i e x i − ϕ performthedivisionoperationinEquation(4). For y ,the (cid:80) p (cid:80) d/p e x ( j ) − ϕ · v ( j ) (4) firstpartialvectorisprocessedsimilarly. However,wefind = j =1 i =1 i i that y 3 − ϕ > b , thentwothreadsareterminatedandthe (cid:80) p (cid:80) d/p e x ( j ) − ϕ j =1 i =1 i firstthreadrecomputesallpartialvectorsaccordingtothe synchronizedpartialsoftmaxschemeinFigure3(b). Theinneraccumulationinboththenumeratorandthede- nominatoronlytakethepartialvectors x ( j ) and v ( j ) asinput,

#### 4 F LAT GEMM O PTIMIZATION WITH

thus they can be processed asynchronously and individu- ally. The outer accumulation is only processed after all D OUBLE B UFFERING partialvectorsareprocessed. AswecanseeinFigure3(c), Motivation. The process of the decode phase is mainly each f ( x ( j ) ) iscalculatedindividually,and softmax ( x ) is composed of GEMV (batch size=1) or flat GEMM calculatedafterall x ( j ) iscalculated. (batch size > 1) operation. Without loss of general- Approach: Recomputation. Withoutlossofgenerality,we ity, GEMV/GEMM operations can be represented using assume a < x i − ϕ < b for each x i to ensure precision M,N,K , where the sizes of two multiplied matrices are and avoid overflow. Then, the partial softmax operation M × K and K × N . Tiling is a common technique for isprocessedindividually. However, when x i − ϕ ≤ a or computing GEMMs on GPUs. The original matrices are

FlashDecoding++:FasterLargeLanguageModelInferencewithAsynchronization,FlatGEMMOptimization,andHeuristics B N B N B N C 1 = A 1 · B 1 + A 2 · B 2 + A 3 · B 3 +…

| B1  | B’1 |     |     |
| --- | --- | --- | --- |
| B2  | B’2 |     |     |
| B3  | B’3 |     |     |
| …   | …   |     |     |
|     |     |     | B   |
| C1  | C2  |     | C   |

32 64 128 256 512 32 64 128 256 512 B C 2 = A 1 · B’ 1 + A 2 · B’ 2 + A 3 · B’ 3 +… K 1024 1024 in shared memory for loading

| √   |       |       |       |      |
| --- | ----- | ----- | ----- | ---- |
| √p  | arall | elism | -boun | ded  |
| √   | √     |       |       |      |
|     |       | √
√   |       |      |
|     |       | √     | mem   | ory  |
|     |       | √     | -bou  | nded |
|     |       | √     |       |      |

| √   |       |       |       |      |
| --- | ----- | ----- | ----- | ---- |
| √p  | arall | elism | -boun | ded  |
| √
√ |       |       |       |      |
|     | √
√   |       |       |      |
|     |       | √     | mem   | ory  |
|     |       | √     | -bou  | nded |
|     |       | √     |       |      |

|     |      |
| --- | ---- |
|     |      |
|     | A1B1 |

2048 2048 in shared memory for computing

| A1B’1 | idle |
| ----- | ---- |

4096 4096 8192 8192 idle N 16384 N 16384 32768 32768 A 1 B 1 A 2 B 2 A 1 B’ 1 A 2 B’ 2 65536 65536 131072 131072 A 3 B 3 A 2 B 2 A 3 B’ 3 A 2 B’ 2

| A1  | A2  | A3  | …   |     | A   |
| --- | --- | --- | --- | --- | --- |

262144 262144 … en il e … M m K =4096 K =12288 GPU Block i T GPU Block √ B N with the best flat GEMM performance for a certain N K N 1 2 Figure7. DoublebufferingforflatGEMMwhen N − dimensionis Figure6. Normalized flat GEMM performance under different large.The M − dimensionispaddedto8andnottiled. N − dimension sizes and N − dimension tiling sizes. We set M =8 andexecuteGEMMontheNVIDIATeslaA100GPU. exposingacontradictiononimprovingtheperformanceof tiledintomultiplesub-matrices,andthendistributedacross GEMV or flat GEMM. We depict the normalized perfor- differentcomputingunitstoenableparallelprocessing. Pre- manceoftheflatGEMMinFigure6withdifferent N and viousLLMinferenceenginesutilizeTensorCoretoacceler- B .Ourkeyinsightis, forthesmaller N ,theflatGEMM N atetheseoperationsusinglibrarieslikecuBLAS(NVIDIA, isparallelism-bounded . Thereare108StreamingMulti- 2017c)andCUTLASS(NVIDIA,2017a).Althoughmodern processors (SMs) in the NVIDIA Tesla A100. N tends TensorCorearchitectures(NVIDIA,2023)processGEMM tobeaconstant( e.g., 128or256),whichisrelatedtothe B N with M =8 ,theselibrariesusuallytilethe M − dimension hardwareparallelism(numberofSMs). Anotherkeyinsight to64tohidememorylatency. However,forGEMVorflat is, forthelarger N ,theflatGEMMbecomesmemory- GEMM operations in the decode phase, we usually have bounded . Theperformanceofthesecasescanbeimproved M ≪ 64 andthe M − dimensionispaddedto64withze- byhidingmemoryaccesslatency. ros. Thepaddingleadstounder-utilizedcomputation,and Approach: Double Buffering. In order to hide memory the key problem is to process GEMV or flat GEMM accesslatency,weintroducethedoublebufferingtechnique. operations with smaller tiles ( i.e., padding to 8 corre- for the flat GEMM operation. We allocate two separate spondingtomodernTensorCorearchitectures)inthe buffers in the shared memory. The tile in one buffer per- M − dimension . formstheGEMMoperation,whileanotherbufferloadsa Challenge. ProcessingGEMVorflatGEMMoperations newtileforthenextGEMMoperation. Thus,thecomputa- isnon-trivialwhenthe M − dimensionispaddedto8. The tionandthememoryaccessareoverlapped. Weapplysuch tilingtechniqueinmodernlibrarieslikecuBLAS(NVIDIA, atechniquewhen N islargeinourpractice. 2017c)andCUTLASS(NVIDIA,2017a)canonlybeap- Example. Figure7showstheexampleofourflatGEMM plied to the N − dimension and the K − dimension. Tiles optimization with double buffering. For M < 8 , the onthe K − dimensionareprocessedsequentiallyinaGPU M − dimensionisfirstpaddedto8consideringmodernTen- blocktoavoidatomicoperationsduringreduction. Tiling sor Core architectures. Workloads in the K − dimension onthe N − dimensionaffectsbothparallelismandcompu- tation/memoryratio,whicharebothimportantforGEMV areprocessedwithinoneGPUblock( e.g., A 1 ,A 2 ,A 3 ,... ), whileworkloadsinthe N − dimensionareprocessedinpar- andflatGEMMacceleration. allelusingdifferentGPUblocks( e.g., C 1 ,C 2 ,... ). Wetake Analysis and Insights. Assume that tiling sizes of the GPU Block as an example, the first tile for each matrix 1 N − dimensionandthe K − dimensionare B N and B K ,re- inthe K − dimension( i.e., A and B )isloadedtotheleft 1 1 spectively.ThecomputationofeachGEMMtileis 2 × M × bufferinthesharedmemory. Then,theGEMMoperationis B N × B K withtotal B = N × K GEMMtiles. Thetotal performedbetween A and B . Consequently, A and B B N × B K 1 1 2 2 memoryaccessis ( M × B K + B N × B K ) × B + M × N . areloadedtotherightbufferinthesharedmemory. Thefol- Thus,thecomputation/memoryratiois: lowingtilesareprocessedsimilarlyaccordingtothedouble bufferingscheme. 2 × M × B N × B K × B ( M × B K + B N × B K ) × B + M × N 2 × M × K (5) 5 H EURISTIC D ATAFLOW WITH = K + M × K + M H ARDWARE R ESOURCE A DAPTION B N Motivation. Although FlashDecoding++ optimizes the Ontheotherhand,theparallelismis N . Thus,thecompu- flat GEMM operation in Section 4, it does not cover all B N tation/memoryratioshowsapositivecorrelationwith B N operations(evenonlyforGEMMs)intheLLMinference. whiletheparallelismshowsanegativecorrelationwith B N , AsmentionedinFigure2(a),theshapesofGEMMsindif-

FlashDecoding++:FasterLargeLanguageModelInferencewithAsynchronization,FlatGEMMOptimization,andHeuristics For a certain LLM, traverse four [N, K] selections

|                                                                                                                            | Operation          | M        |     | N    | K   |
| -------------------------------------------------------------------------------------------------------------------------- | ------------------ | -------- | --- | ---- | --- |
| Prefill
phase                                                                                                              | K, Q, V projection | SeqLen*B |     | HD*3 | HD  |
|                                                                                                                            | O projection       | SeqLen*B |     | HD   | HD  |
|                                                                                                                            | FFN1               | SeqLen*B |     | FD   | HD  |
|                                                                                                                            | FFN2               | SeqLen*B |     | HD   | FD  |
| Decode
phase                                                                                                               | K, Q, V projection | B        |     | HD*3 | HD  |
|                                                                                                                            | O projection       | B        |     | HD   | HD  |
|                                                                                                                            | FFN1               | B        |     | FD   | HD  |
|                                                                                                                            | FFN2               | B        |     | HD   | FD  |
| HD: Hidden dimension size Only 4 shapes
FD: Dimension size after the first FFN
B: Batch size
SeqLen: Input sequence length |                    |          |     |      |     |

| ……   | U         | sing cuBLAS   | /CUTLASS…        |          |
| ---- | --------- | ------------- | ---------------- | -------- |
| M=17 |           |               |                  |          |
| M=16 |           | M2            | M2               |          |
| ……   |           |               |                  |          |
| M=9  |           |               |                  |          |
| M=8  | M2        |               |                  | M2       |
| ……   | Usin      | g our flat GE | MM optimizat     | ion      |
| M=3  |           |               |                  |          |
| M=2  | M         | M1            |                  | M1       |
| M=1  | Usin1g GE | MV on CUDA    | Core M (1e.g., F | astGEMV) |

M=1 Impl.B > Impl.A? M++ Find M 1 Impl.C > Impl.B? M++ Find M 2 K, Q, V O FFN 1 FFN 2 projection projection ImplA = FastGEMV ImplB = our flat GEMM End [N, K] = [N, K] = [N, K] = [N, K] = ImplC = CUTLASS [12288, 4096] [4096, 4096] [11008, 4096] [4096, 11008] (a) Different shapes of GEMMs in LLM (b) Decision flow (c) Example of heuristic dataflow with hardware resource adaption Figure8. Heuristicdataflowwithhardwareresourceadaptionin FlashDecoding++ .(a)Onlyfour [ N,K ] shapesexistforacertainLLM. (b)Thedecisionflow. Wetraverseall [ N,K ] selectionsandprofiletheperformanceofthreerepresentativeimplementations. M is increasedtofindtwoinflectionpointsforruntimeheuristicdataflow.(c) FlashDecoding++ heuristicallyutilizesTensorCore/CUDACore withthecorrespondingGEMV/GEMMimplementationbyreferringtoalookuptable. ferentoperationsandtwophasesvary. Thus,theGEMM impactsthekernelperformance. Alltheseinfluentialfactors workloadintheLLMinferencecanbeGEMV(batchsize=1 buildalargesearchspace,makingitnon-trivialtogenerate for the decode phase), flat GEMM (small batch size for aneffectivemappingbetweenthelinearworkloadandthe the decode phaseandshortsequencelengthforthe prefill correspondingoptimalimplementation. phase)andconventionalGEMM(largebatchsizeorlong AnalysisandInsights. Althoughallinfluentialfactorsform sequencelengthforthe prefill phase). Inordertoleverage a large search space, the homogeneity of different layers the powerful computational ability of Tensor Core, cur- inLLMsignificantlyreducesthesearchspaceforoperator rentframeworkslikeFasterTransformer(NVIDIA,2017b) optimization. Figure2(a)showsfourlinearGEMV/GEMM andDeepSpeed(Aminabadietal.,2022)tendtoutilizethe operations in the prefill phase and the decode phase, i.e., highly optimized GEMM implementation from cuBLAS K,Q,V projection, O projection,andtwofeedforwardop- (NVIDIA,2017c)todealwithdifferentworkloads. How- erations. Each GEMV/GEMM operation can be can be ever,theTensorCoreimplementationfailswiththeGEMV abstractedasamultiplicationbetweenan( M × K )-shaped workload. TheGEMVworkloadcanbeoptimizedbyutiliz- matrixanda( K × N )-shapedmatrix. Ourkeyinsightis, ingCUDACoreinpreviousdesignslikeFastGEMV(Wang, there are only four [ K,N ] shapes for a certain LLM. 2023).ForaLlama2-7Blinearlayerinthe decode phase,the Moreover, M isonlyrelatedtotheinputsequencelength TensorCoreimplementationfromcuBLASonlyachieves andthebatchsizeforthe prefill phase,andthebatchsize 82.15%oftheperformanceofCUDACoreimplementation forthe decode phase. Figure8(a)showslimitedshapesof usingFastGEMVonanNVIDIAA100GPU.Ontheother GEMV/GEMMoperationsintheLLMinference. hand, using CUDA Core to do the projection on a batch- size=4decodinginputonlyachieves49.75%performance Approach: Decisionflowforinflectionpoints. Because comparedwiththeTensorCoreimplementation. Thus,in only four [ K,N ] shapes exist for a certain LLM, we use ordertoapproachtheoptimalcomputationperformance, a threetypesofimplementationsforGEMV/GEMMopera- heuristicdataflowissupposedtobeexploitedfordiffer- tionswhen M varies: FastGEMVfortheGEMVandflat entworkloads. GEMMoperations(ImplA),ourflatGEMMoptimization inSection4(ImplB),andtheCUTLASS(NVIDIA,2017a) Challenge. Althoughaheuristicdataflowpotentiallyexists libraries optimized for the conventional GEMM (ImplC). in the implementation of different linear workloads, it is Thus,itisimportanttodecidewhetherapplyingImplAor challengingtobuildthemappingfromacertainworkload ImplBforasmall M ,andImplBorImplCforalarge M . toanoptimalimplementation. InthescenarioofLLMinfer- Figure 8(b) shows the decision flow. FlashDecoding++ ence,therearevariousfactorsthatinfluencetheimplemen- profilestheperformanceofImplAandImplBforacertain tationperformanceoflinearworkloads: (a)Inputdynamics. M ,andincreases M tofindaninflectionpoint M 1 where Thevarietyofthebatchsizeandtheinputsequencelength theperformanceofImplBisbetterthanImplA.Anotherin- bringsdynamicworkloads. (b)Modeldiversity. Thelinear flectionpoint M 2 isfoundsimilarlywheretheperformance workloadvarieswithdifferentmodelstructuresandsizes. ofImplCisbetterthanImplB.Notethateach [ N,K ] gets (c)GPUcapacities. Therelativeperformancebetweenim- itsindividual M 1 and M 2 . plementationschangeswithGPUcharacteristics, suchas memorybandwidth,cachesize,andcomputationalability. Approach: Heuristic dataflow. For the runtime LLM (d)Engineeringeffects. Theengineeringeffortalsohighly inference, FlashDecoding++ adoptsImplAusingCUDA

| FlashDecoding++:FasterLargeLanguageModelInferencewithAsynchronization,FlatGEMMOptimization,andHeuristic                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| HF vLLM DeepSpeed TensorRT-LLM ppl FlashDecoding Ours Ours (token/s)
1.0 1000 1.0 600 B7-2amalL)a(
Throughput
pudeepS 400
0.5 500 0.5
200
0.0 0 0.0 0
128 1k 8k 32k 128 1k 8k 32k 128 1k 8k 128 1k 8k 128 1k 2k 4k 128 1k 2k 4k 128 1k 2k 128 1k 2k
1.0 1000 1.0 600 B7.6-TPO)b(
Throughput
pudeepS 400
0.5 500 0.5
200
0.0 0 0.0 0
128 1k 8k 32k 128 1k 8k 32k 128 1k 8k 128 1k 8k 128 1k 2k 4k 128 1k 2k 4k 128 1k 2k 128 1k 2k
1.0 1000 1.0 600 B6-2MLGtahC)c(
Throughput
pudeepS
400
0.5 500 0.5
200
0.0 0 0.0 0
128 1k 8k 32k 128 1k 8k 32k 128 1k 8k 128 1k 8k 128 1k 2k 4k 128 1k 2k 4k 128 1k 2k 128 1k 2k
batchsize=1 batchsize=2 batchsize=4 batchsize=8 batchsize=1 batchsize=2 batchsize=4 batchsize=8
NVIDIATeslaA100 NVIDIARTX3090
Figure9.SpeedupofthedecodephaseonNVIDIAGPUs,normalizedtoFlashDecoding++.Blankbarsrepresentthemodelcannotb
executed:(1)HuggingFaceandDeepSpeedrunoutofmemorywithlongsequences.(2)vLLMdoesnotsupportOPT-6.7Bwithsequenc
ength>2kandChatGLM2-6B.(3)TensorRT-LLMfailstocompileforOPT-6.7BandChatGLM2-6Bwithsequencelength>=8k.(4
FlashDecodingandpplonlysupportsLlama2models.
Table1. HardwarePlatforms Table2. ModelConfiguration
NVIDIA AMD Context
Model Dimension Heads Layers
Length
TeslaA100 RTX3090 MI210 RX7900XTX
GPU 80GB 24GB 64GB 24GB Llama2-7B 4096 32 32 4k
CUDA12.1 CUDA11.6 ROCm5.7 ROCm5.6 Llama2-13B 5120 40 40 4k
OPT-6.7B 4096 32 32 2k
IntelXeon IntelXeon AMDEPYC IntelCore
ChatGLM2-6B 4096 32 32 32k
CPU Silver8358P Gold6226R 7K62 i9-10940X
2.60GHz 2.90GHz 2.60GHz 3.30GHz
6.1.1 HardwarePlatforms |

FlashDecoding++:FasterLargeLanguageModelInferencewithAsynchronization,FlatGEMMOptimization,andHeuristics

|     |     |     |                 |     |     |         |     |     |
| --- | --- | --- | --------------- | --- | --- | ------- | --- | --- |
|     |     |     | 128 1k 2k 4k 12 |     |     | 8 1k 2k |     |     |
|     |     |     |                 |     |     |         |     |     |
|     |     |     | 128 1k 2k 4k 12 |     |     | 8 1k 2k |     |     |
|     |     |     |                 |     |     |         |     |     |
|     |     |     |                 |     |     |         |     |     |

Figure9. .Blankbarsrepresentthemodelcannotbe executed:(1)HuggingFaceandDeepSpeedrunoutofmemorywithlongsequences.(2)vLLMdoesnotsupportOPT-6.7Bwithsequence 8k.(4)

Corewhen M <M 1 ,andImplB/ImplCusingTensorCore We evaluate the performance of FlashDecoding++ and when M 1 ≤ M < M 2 / M 2 ≤ M . Notethatthedecision otherLLMenginesonbothNVIDIAandAMDplatforms flowareexecutedoffline,itdoesnotaffecttheperformance tomakeacomprehensivecomparison. Wechoosetwodif- ofruntimeLLMinference. ferentGPUsforeachplatform: TeslaA100andRTX3090 forNVIDIA,MI210andRX7900XTXforAMD.Weshow Example. Figure8(c)showsanexampleofapplyingthe thedetailedconfigurationinTable1. heuristicdataflowfortheLlama2-7Bmodel. Four [ N,K ] shapes are [12288, 4096] for K,Q,V projection, [4096, 6.1.2 LLMEngineBaselines 4096]for O projection,[11008,4096]and[4096,11008]for FFN.Foreach [ N,K ] ,theinflectionpointsarefoundbased We implement our FlashDecoding++ using the Pytorch- onthedecisionflowinFigure8(c). Then,alookuptable based front-end with the C++ and CUDA backend for isformed,andeachGEMV/GEMMoperationisexecuted NVIDIA GPUs while ROCm for AMD GPUs. We com- accordingtocorrespondingimplementationsduringruntime. pare the inference performance in both prefill phase and In this example, FastGEMV is adopted for the K,Q,V decode phase with the following LLM engine baselines: projection when batch size=1 ( M = 1 ) for the decode Hugging Face (HF) v4.34.1 (Wolf et al., 2020), vLLM phase, and our flat GEMM optimization is applied when v0.1.7(Kwonetal.,2023),DeepSpeedv0.11.1(Aminabadi batchsize=1/inputsequencelength=8forFFN 1 ( M =8 ). etal.,2022),TensorRT-LLMv0.5.0(Vaidyaetal.,2023), OpenPPL(Sensetime,2023a),FlashAttentionv2.3.5(Dao,

#### 6 E VALUATION 2023)andFlashDecoding(Daoetal.,2023). Thesebase-

linesareintroducedinSection7. 6.1 ExperimentsSetup 6.1.3 Models Weevaluatetheperformanceof FlashDecoding++ ondif- ferent GPUs with various Large Language Models. We We evaluate the performance of FlashDecoding++ with comparetheperformancewithseveralstate-of-the-artLLM otherLLMinferenceenginesonthreetypicalLargeLan- inferenceengines. guage Models: Llama2, OPT, and ChatGLM2. Table 2

| hAsynchronization,FlatGEMMOptimization,andHeuristics
HF Ours Ours (token/s)
B                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |                     |        |                                                                                                                                                                                                           |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------- | ------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| HF Ours Ours (token/s)
B                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |                     |        |                                                                                                                                                                                                           |
|                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          | Ours Ours (token/s) |        |                                                                                                                                                                                                           |
| 3 7-2amalL)a(
2 pudeepS
1
0
128 1k 2k 4k 128 1k 2k 128 1k 2k 128
3 B7.6-TPO)b(
2 pudeepS
1
0
128 1k 2k 4k 128 1k 2k 128 1k 2k 128
batchsize=1 batchsize=2 batchsize=4 batc
Figure11.SpeedupofthedecodephaseonAMDRX
HF vLLM Ours Ours (token/s)
5 B7-2amalL)a(
4 pudeepS
HF Ours Ours (token/s) 3
3 2
1
pudeepS
2 0
128 1k 2k 4k 128 1k 4k 128 1k 4k 128 1k
1 4 B31-2amalL)b(
0 3 pudeepS
128 512 1k 2k 128 512 1
2
(a)Llama2-7B (b)Llama2-13B 1
0
128 1k 2k 4k 128 1k 4k 128 1k 2k 128 1k
5 B7.6-TPO)c(
4 pudeepS
3
2
1
0
128 1k 2k 4k 128 1k 4k 128 1k 4k 128 1k
batchsize=1 batchsize=2 batchsize=4 batchsiz
Figure12.SpeedupofthedecodephaseonAMDMI
are blank bars for vLLM because it doesn’t suppor |                     |        | 400
300 Throughput
200
100
0
1k
400
300 Throughput
200
100
0
1k
hsize=8
7900XTX.
600
Throughput
400
100 200
0
50 2k
300
0 Throughput
k 200
100
0
2k
600
Throughput
400
200
0
4k
e=8
210. There
t sequence |
|                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |                     |        |                                                                                                                                                                                                           |
|                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |                     |        |                                                                                                                                                                                                           |
|                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          | 128 1k 2k 128 1k    | 2k 128 |                                                                                                                                                                                                           |
|                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |                     |        |                                                                                                                                                                                                           |
|                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |                     |        |                                                                                                                                                                                                           |
| lengthover2kforOPT-6.7B.
decode phase, FlashDecoding++ achieves up to 4.86×
speedup compared with Hugging Face implementations
on three LLMs and two GPUs. The average speedup
over vLLM, DeepSpHFeedO,uTrsenOsourrs R(toTke-nL/sL) M, OpenPPL, and
3 100
FlashDecodingis1.24×,1.44×,1.13×,1.24×,and1.21×
2 pudeepS
50 (1.37×onTeslaA100comparedwithFlashDecoding),re-
1
spectively.Fortheprefillphase,FlashDecoding++achieves
0 0
128 512 1k 2k 128 512 1k up to 1.40× speedup compared with Hugging Face im-
plementations.(a)TLlhamea2a-v7Berage speed(bu)pLlaomva2e-r13BDeepSpeed,
TensorRT-LLM,OpenPPL,FlashAttention2andFlashDe-                                                                   |                     |        |                                                                                                                                                                                                           |

FlashDecoding++:FasterLargeLanguageModelInferencewithAsynchronization,FlatGEMMOptimization,andHeuristics

HF TensorRT-LLM DeepSpeed PPL FlashAttention2 Ours Ours-FTL batchsize= 1 2 4 8 *Ours-FTL:Ours(firsttokenlatency[ms]) 2.0 2.E+04 pudeep 1.E+04 La 1.0 t en S 5.E+03 cy 0.0 0.E+00 1k 8k 32k 1k 8k 32k 1k 8k 32k 1k 8k (a)Llama2-7B@A100 1.5 2.E+04 pudeep 1.0 1.E+04 La t en 0.5 5.E+03 cy S 0.0 0.E+00 1k 8k 32k 1k 8k 32k 1k 8k 1k 8k (b)Llama2-13B@A100 1.5 6.E+03 pudeep 1.0 4.E+03 La t en 0.5 2.E+03 cy S

| Ours Ours (token/s) |         |
| ------------------- | ------- |
| 28 1k 4k 128 1k 4k  | 128 1k  |
| 1k 2k 128 5         | 12 1    |
| 2-7B (b)Lla         | ma2-13B |
| 28 1k 4k 128 1k 2k  | 128 1k  |
|                     |         |
|                     |         |
|                     |         |

0.0 0.E+00 1k 8k 32k 1k 8k 1k 8k 1k 8k (c)ChatGLM2-6B@A100 1.5 4.E+03 pu 1.0 3.E+03 La deep 2.E+03 t en S 0.5 1.E+03 cy 0.0 0.E+00 1k 8k 1k 8k 1k 1k (d)Llama2-7B@3090 1.5 4.E+03 pudeep 1.0 3.E+03 La 2.E+03 t en S 0.5 1.E+03 cy 0.0 0.E+00 1k 8k 1k 8k 1k 1k (e)ChatGLM2-6B@3090 Figure10. Speedupofthe prefill phaseonNVIDIAGPUs,normal- izedtoFlashAttention.Blankbarsrepresentfailedexecution:(1) HuggingFace,DeepSpeedandTensorRT-LLMrunoutofmemory withlongsequences.(2)vLLMdoesnotsupportChatGLM2-6B. (3)TensorRT-LLMfailstocompileonRTX3090GPUswith24GB memory,andfailstocompileforChatGLM2-6Bwithsequence length > = 8k.(4)pplonlysupportsLlama2models. showsthedetailedconfigurationofthesemodels. Notethat

| ×,1.44×,1.13×,1.24×, |        |
| -------------------- | ------ |
| comparedwithFla      | shDec  |
| llphase,FlashDec     | oding+ |

theremaybeseveralmodelsinoneLLM( e.g., Llama2-7B, Llama2-13B)withdifferentconfigurations( e.g., numberof headsandlayers).

• Llama2 (Touvronetal.,2023)isamainstreamopen- source LLM set released by Meta in 2023. It is a collectionofpretrainedandfine-tunedgenerativetext modelsranginginscalefrom7Bto70Bparameters. • OPT (Zhangetal.,2022),isasuiteofdecoder-only pre-trainedtransformersrangingfrom125Mto175B parametersreleasedbyMetaAI. • ChatGLM2 (Duetal.,2022)isanopen-sourceLLM supportingbilingual(Chinese-English)chat.

6.2 ComparisonwithState-of-the-art Wecompare FlashDecoding++ withstate-of-the-artLLM inference engines in Figure 9 and Figure 10 on NVIDIA GPUs,Figure11andFigure12forAMDGPUs. Forthe

codingis1.05 × ,1.06 × ,1.08 × ,1.09 × ,and1.08 × ,respec- tively. For prefill phase, FlashDecoding++ performsworse thansomebaselineswithshortsequencesbutalwaysgains speedupwithlongsequences. Thereasonisthat,for prefill phase,weonlyoptimizetheattentionoperation,andtheat- tentionoperationoccupiesmoreofthelatencyassequence lengthgrows. Wealsoshowthe decode resultsontwoAMDGPUs. Cur- rently,onlyHuggingFaceandvLLMcanbeexecutedon AMDGPUsasthebaselines,andvLLMdoesnotsupport RX7900XTXyet. FlashDecoding++ achievesupto2.41 × and 4.35 × comparedwithHuggingFaceonRX7900XTX andMI210,respectively. AndonMI210,theaveragespeed of FlashDecoding++ comparedtovLLMis1.86 × .

FlashDecoding++:FasterLargeLanguageModelInferencewithAsynchronization,FlatGEMMOptimization,andHeuristics

FlashAttention2 Ours 16384]

| xformers FlashAt
0903XTR
6
pudeepS
4
2
)a(
0
128 1k 8k 32k128 1k 8k 32k
4 001Aa
3 pudee
2 |
| ----------------------------------------------------------------------------------------- |
| 1 lseT)b( pS
0
128 1k 8k 32k128 1k 8k 32k
batchsize=1 batchsize=2                         |

| [4096, 4096] [4096, 11008] [4096, 12288] [12288, 4096] [4096, 163
[16384, 4096] [4096, 13696] [13696, 4096] GeoMean [K,
1.6 1.4
Tesla A100 RTX 3090
1.4 pudeep 1.2
1.2 1 | [4096, 4096] [4096, 11008] [4096, 12288] [12288, 4096] [4096, 163
[16384, 4096] [4096, 13696] [13696, 4096] GeoMean [K, |          |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------- | -------- |
|                                                                                                                                                                          |                                                                                                                         | RTX 3090 |

N]

32k128 1k 8k 32k128 1k 8k 32k 0.8 1 1 2 4 8 16 1 2 4 8 16 M Figure15. SpeedupovercuBLASwithflatGEMMoptimization.

be applied in specific domains. For both models, the in- 32k128 1k 8k 32k128 1k 8k 32k batchsize=4 batchsize=8 putstothesoftmaxoperationareobtainedthroughmultiple Figure13. Benefitsofasynchronizedsoftmax( prefill phase). datasets.99%ofthesoftmaxinputinCodeLlama-7Branges from -0.25 to 17.6, while that of Vicuna-7B ranges from HF FlashDecoding xformers xformers-decoder Ours Ours (w/o async) -0.8to9.8. Thus,theasynchronizedsoftmaxmethodisalso 8 applicabletothosefine-tunedmodels. pudeep 6 @TeslaA100GPU @RTX3090GPU 4 S 2 6.3.2 FlatGEMMOptimization cache 0 64 1024 2048 64 1024 2048 Geo 64 1024 2048 64 1024 2048 Geo length batchsize=1 batchsize=4 Mean batchsize=1 batchsize=4 Mean Benefits. WetestourflatGEMMkernelperformancewith 1.5 state-of-the-artGEMMlibrary,cuBLASontwoNVIDIA pudeep 1

|     |
| --- |
|     |

GPUs. The version of cuBLAS is CUDA 11.8. We vary 0.5 S M from1to16todemonstratetheflatGEMMoperation cache 0 4k 32k 64k 4k 32k 64k Mean Geo 4k 32k 64k 4k 32k 64k Mean Geo inLLMinference,andeight [ K,N ] configurationsusedin length batchsize=1 batchsize=4 batchsize=1 batchsize=4 threeLLMs(Llama2-7B,OPT-6.7B,andChatGLM2-6B) Figure14. Benefitsofasynchronizedsoftmax( decode phase). aredepictedinFigure15. TheflatGEMMoptimizationin FlashDecoding++ achievesanaverageof7%and17%(up 6.3 AblationStudies to52%)speeduponTeslaA100andRTX3090,respectively. LibrariesincludingcuBLASaredesignedforgeneralpur- 6.3.1 AsynchronizedSoftmaxComputation pose,hencenotthebestfortheflatGEMMpractice. The Benefits. The asynchronized softmax scheme can be ap- speedupis9%and23%forsmall M ( i.e., 1and2),show- pliedtoboththe prefill phaseandthe decode phase. Wetest ingthattheproposedflatGEMMoptimizationexploresthe theproposedschemeagainststate-of-the-artattentionimple- computationcapabilitywithsmallbatchsizes. mentationsinFigure13andFigure14onNVIDIAGPUs. Scalability. Theusageofdoublebufferingwithlargesizein Forthe prefill phase, FlashDecoding++ achieves1.52 × and N -dimensionislimitedbythesharedmemory(L1cache) 1.19 × averagespeedupcomparedwithxformers(Lefaudeux size of GPUs. The results in Figure 15 demonstrate that et al., 2022) and FlashAttetion2. For the decode phase, the strategy works with both NVIDIA Tesla A100 GPUs FlashDecoding++ outperformsthedecoding-tailored im- (192KBL1cacheperSM)andNVIDIARTX3090GPUs plementationofxformers(denotedasxformers-decoderin (128KBL1cacheperSM)thankstothelargeL1datacache. Figure14)withshortKVcachelength,andachievesupto ButforAMDGPUs,doublebufferingfailstobenefittheflat 2.02 × speedupoverFlashDecodingwithlongcontext. GEMMperformanceduetoalimitedL1datacache(16KB Correctness. Theabsolutedifferencebetweentheproposed perCUforAMDMI210). Withoutdoublebuffering, the attention method and PyTorch is average 99.7% < 1e-2, flatGEMMoptimizationperformsbadlyinmanycases. In andall < 1e-1(FlashAttentionleadsto99.8% < 1e-2v.s. fact,onAMDGPUs,wesignificantlyrelyonheuristicsto PyTorch). AsmentionedinSec.3,weintroducearecom- achieveperformancegains. putationmechanismintotheasynchronizedsoftmax,which automaticallyselectsFlashAttentionforcomputationwhen 6.3.3 BenefitsofHeuristicDataflow theintermediateresultsoverflow. Thefrequencyofrecom- We test speedup of the decode phase by adopting the putation is statistically obtained to be 0.45% on average heuristicdataflowinthreeLLMs(Llama2-7B,OPT-6.7B, across datasets including ARC (Clark et al., 2018), Hel- and ChatGLM2-6B) on NVIDIA GPUs, and two LLMs laSwag(Zellersetal.,2019)andWinogrande(Sakaguchi (Llama2-7B,OPT-6.7B)onAMDGPUs. Theinputlength etal.,2019). issetto1024,andtheresultsareshowninFigure16. The Scalability. Weextendourapproachtomodelsincluding heuristicdataflowachievesanaverageof10%and20%(up CodeLlama-7B(Rozie`reetal.,2023)andVicuna-7B(Chi- to29%)speeduponTeslaA100andRTX3090,respectively. ang et al., 2023), which are fine-tuned on Llama2-7B to OnAMDGPUs,theextensionofFastGEMVimplementa-

FlashDecoding++:FasterLargeLanguageModelInferencewithAsynchronization,FlatGEMMOptimization,andHeuristics

| Llama2-7B OPT-6.7B ChatGLM2-6B
1.2 1.4
Tesla A100 RTX 3090
1.3
pud                                                                                                   |               |            |          |          |     |          |        |            |     |     |     |       |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------- | ---------- | -------- | -------- | --- | -------- | ------ | ---------- | --- | --- | --- | ----- |
|                                                                                                                                                                      |               |            |          |          |     | RTX 3090 |        |            |     |     |     |       |
| eepS 1.1 1.2
1.1
1 1
1 2 4 8 16 1 2 4 8 16                                                                                                                           |               |            |          |          |     |          |        |            |     |     |     |       |
| batch size batch size                                                                                                                                                |               |            |          |          |     |          |        |            |     |     |     |       |
| 2 2.6
1.8 MI210 2.2 RX7900
pudeepS
1.6
1.8
1.4
1.2 1.4 Hugging Face DeepSpeed Ours Ours (token/s)
1 1 4 90 pudeepS
1 2 4 8 1 2 4 8 3 85
batch size 2 batch size
1 80 |               |            |          |          |     |          |        |            |     |     |     |       |
|                                                                                                                                                                      | MI21          |            |          |          |     |          | RX7900 |            |     |     |     |       |
|                                                                                                                                                                      |               |            |          |          |     |          |        |            |     |     |     |       |
|                                                                                                                                                                      |               | Hugging F  | ace Deep | Spee1d.4 |     | Our      | s Ou   | rs (token  | /s) | 90  |     |       |
|                                                                                                                                                                      | 1             |            |          |          |     |          |        |            |     |     |     | 85
80 |
|                                                                                                                                                                      | 1 2 4 8 1 2 4 |            |          |          |     |          |        |            |     |     | 8   |       |
|                                                                                                                                                                      |               | batch size |          |          |     |          |        | batch size |     |     |     |       |
|                                                                                                                                                                      |               |            |          |          |     |          |        |            |     |     |     |       |

| Tesla A10 |     |     |
| --------- | --- | --- |
|           |     |     |

Figure16. 0 Benefitsoftheheuristicdataflow(inputlength=1024). 75 128 512 1k 2k 4k

Hugging Face DeepSpeed Ours Ours (token/s) 4 90

|     |     |     |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|     |     |     |     |     |     |     |     |     |     |     |
|     |     |     |     |     |     |     |     |     |     |     |
|     |     |     |     |     |     |     |     |     |     |     |

pudeep 3 85 2 1 80 S 0 75 128 512 1k 2k 4k Figure17. PerformanceonLlama2-13BontwoTeslaA100GPUs. tionprovestobehighlyefficient, andleadstosignificant performance gains with small batch sizes. The average speedupofusingheuristicsis57%and37%onMI210and RX7900XTX,respectively.

6.4 Multi-GPUsPerformance FlashDecoding++ supportsexecutinglargeLLMsonmul- tiple GPUs. We use Llama2-13B running on 2 NVIDIA TeslaA100GPUstoevaluatetheperformanceof FlashDe- coding++ . The result in Figure 17 shows that, FlashDe- coding++ achieves2.48 × and1.19 × higher decode phase throughputcomparedwithHuggingFace(Wolfetal.,2020) andDeepSpeed(Aminabadietal.,2022).

#### 7 R ELATED W ORKS

Large language model inference acceleration has gained significantattentioninrecentresearch,withseveralnotable approaches and techniques emerging in the field. Deep- Speed (Aminabadietal.,2022)isacomprehensiveengine that optimizes both the training and inference phases for LLMs. Itachievesrobustinferenceperformancethrough kernelfusionandefficientGPUmemorymanagement,with a particular focus on optimizing memory usage for KV- cache. vLLM (Kwonetal.,2023)improvesGPUmemory utilizationbyefficientmemorymanagementtechniquesand thePageAttentionmethod,leadingtoincreasedmaximum batchsizesandelevatingtheupperlimitofinferenceper- formance. FlashAttention (Daoetal.,2022;Dao,2023) optimizestheself-attentioncomputationprocessduringthe prefill phase through improved parallelism and workload distribution. FlashDecoding (Daoetal.,2023)isanexten-

sionofFlashAttentionandenhancestheparallelismthrough spliting K and V ,supportingefficientself-attentioncompu- tationforlongsequenceduringthedecodephase. Faster- Transformer (NVIDIA,2017b)and OpenPPL (Sensetime, 2023a) implement large model inference engines using C++ to reduce overhead resulting from kernels schedul- ing,comparedto Python implementations. Theyalsoem- ploymemorymanagementtechniquesandkernelfusionto achieveefficientLLMinference. TensorRT-LLM (Vaidya etal.,2023)isbuiltuponthe TensorRT (NVIDIA)andthe FasterTransformer (NVIDIA,2017b)engine( C++ )andin- corporatescutting-edgeopen-sourcetechnologiessuchas FlashAttention (Daoetal.,2022;Dao,2023). Additionally, itenhancesitseaseofusebyprovidingthe PythonAPI .

#### 8 C ONCLUSION

We propose FlashDecoding++ , a fast Large Language Model inference engine in this paper. FlashDecoding++ acceleratesmainstreamLLMswithmultiplehardwareback- end support. FlashDecoding++ proposes three novel de- signs: theasynchronizedsoftmaxwithunifiedmaxvalue, the flat GEMM optimization with double buffering, and the heuristic dataflow with hardware resource adaption, achieving up to 4.86 × and 4.35 × speedup on NVIDIA andAMDGPUscomparedwithHuggingFaceimplementa- tions. FlashDecoding++ alsoachievesanaverageof 1.37 × speedupcomparedwithstate-of-the-artLLMinferenceen- gines,FlashDecoding,onvariousLLMs.

R EFERENCES Text generation inference: Fast inference optimize for llms. [Online], 2023. https://github.com/huggingface/ text-generation-inference/ . Mlcllm: Machinelearningcompilationforlargelanguage models. [Online], 2023. https://github.com/ mlc-ai/mlc-llm . AMD. Toolstotranslatecudasourcecodeintoportablehip c++automatically. [Online],2023. https://github. com/ROCm-Developer-Tools/HIPIFY . Aminabadi, R. Y., Rajbhandari, S., Awan, A. A., Li, C., Li, D., Zheng, E., Ruwase, O., Smith, S., Zhang, M., Rasley,J.,etal. Deepspeed-inference: enablingefficient inferenceoftransformermodelsatunprecedentedscale. In SC22:InternationalConferenceforHighPerformance Computing,Networking,StorageandAnalysis ,pp.1–15. IEEE,2022. Anil,R.,Dai,A.M.,Firat,O.,Johnson,M.,Lepikhin,D., Passos,A.,Shakeri,S.,Taropa,E.,Bailey,P.,Chen,Z.,

FlashDecoding++:FasterLargeLanguageModelInferencewithAsynchronization,FlatGEMMOptimization,andHeuristics Chu, E., Clark, J. H., Shafey, L. E., Huang, Y., Meier- Hellstern, K., Mishra, G., Moreira, E., Omernick, M., Robinson,K.,Ruder,S.,Tay,Y.,Xiao,K.,Xu,Y.,Zhang, Y.,Abrego,G.H.,Ahn,J.,Austin,J.,Barham,P.,Botha, J., Bradbury, J., Brahma, S., Brooks, K., Catasta, M., Cheng,Y.,Cherry,C.,Choquette-Choo,C.A.,Chowd- hery, A., Crepy, C., Dave, S., Dehghani, M., Dev, S., Devlin,J.,D´ıaz,M.,Du,N.,Dyer,E.,Feinberg,V.,Feng, F., Fienber, V., Freitag, M., Garcia, X., Gehrmann, S., Gonzalez,L.,Gur-Ari,G.,Hand,S.,Hashemi,H.,Hou, L.,Howland,J.,Hu,A.,Hui,J.,Hurwitz,J.,Isard,M.,It- tycheriah,A.,Jagielski,M.,Jia,W.,Kenealy,K.,Krikun, M.,Kudugunta,S.,Lan,C.,Lee,K.,Lee,B.,Li,E.,Li, M.,Li,W.,Li,Y.,Li,J.,Lim,H.,Lin,H.,Liu,Z.,Liu, F.,Maggioni,M.,Mahendru,A.,Maynez,J.,Misra,V., Moussalem,M.,Nado,Z.,Nham,J.,Ni,E.,Nystrom,A., Parrish,A.,Pellat,M.,Polacek,M.,Polozov,A.,Pope, R.,Qiao,S.,Reif,E.,Richter,B.,Riley,P.,Ros,A.C., Roy, A., Saeta, B., Samuel, R., Shelby, R., Slone, A., Smilkov,D.,So,D.R.,Sohn,D.,Tokumine,S.,Valter, D., Vasudevan, V., Vodrahalli, K., Wang, X., Wang, P., Wang,Z.,Wang,T.,Wieting,J.,Wu,Y.,Xu,K.,Xu,Y., Xue, L., Yin, P., Yu, J., Zhang, Q., Zheng, S., Zheng, C.,Zhou,W.,Zhou,D.,Petrov,S.,andWu,Y. Palm2 technicalreport,2023. Bridle,J. Trainingstochasticmodelrecognitionalgorithms as networks can lead to maximum mutual information estimationofparameters. Advancesinneuralinformation processingsystems ,2,1989. Chiang, W.-L., Li, Z., Lin, Z., Sheng, Y., Wu, Z., Zhang, H.,Zheng,L.,Zhuang,S.,Zhuang,Y.,Gonzalez,J.E., Stoica, I., and Xing, E. P. Vicuna: An open-source chatbot impressing gpt-4 with 90%* chatgpt quality, March 2023. URL https://lmsys.org/blog/ 2023-03-30-vicuna/ . Clark,P.,Cowhey,I.,Etzioni,O.,Khot,T.,Sabharwal,A., Schoenick, C., andTafjord, O. Thinkyouhavesolved questionanswering? tryarc,theai2reasoningchallenge. ArXiv ,abs/1803.05457,2018. Clusmann, J., Kolbinger, F. R., Muti, H. S., Carrero, Z. I., Eckardt, J.-N., Laleh, N. G., Lo¨ffler, C. M. L., Schwarzkopf, S.-C., Unger, M., Veldhuizen, G. P., et al. The future landscape of large language models inmedicine. CommunicationsMedicine ,3(1):141,2023. Cui, C., Ma, Y., Cao, X., Ye, W., and Wang, Z. Re- ceive, reason, and react: Drive as you say with large languagemodelsinautonomousvehicles. arXivpreprint arXiv:2310.08034 ,2023. Dai, Z., Yang, Z., Yang, Y., Carbonell, J., Le, Q. V., and Salakhutdinov, R. Transformer-xl: Attentive language

models beyond a fixed-length context. arXiv preprint arXiv:1901.02860 ,2019. Dao, T. Flashattention-2: Faster attention with bet- ter parallelism and work partitioning. arXiv preprint arXiv:2307.08691 ,2023. Dao,T.,Fu,D.,Ermon,S.,Rudra,A.,andRe´,C. Flashat- tention: Fastandmemory-efficientexactattentionwith io-awareness. AdvancesinNeuralInformationProcess- ingSystems ,35:16344–16359,2022. Dao, T., Haziza, D., Massa, F., and Sizov, G. Flash-decoding for long-context inference. [Online], 2023. https://crfm.stanford.edu/2023/ 10/12/flashdecoding.html . Dong, Z., Tang, T., Li, L., and Zhao, W. A survey on longtextmodelingwithtransformers.arxiv2023. arXiv preprintarXiv:2302.14502 . Du,Z.,Qian,Y.,Liu,X.,Ding,M.,Qiu,J.,Yang,Z.,and Tang,J. Glm: Generallanguagemodelpretrainingwith autoregressiveblankinfilling. In Proceedingsofthe60th Annual Meeting of the Association for Computational Linguistics(Volume1: LongPapers) ,pp.320–335,2022. DYLAN PATEL, A. A. The inference cost of search disruption-largelanguagemodelcostanalysis. [Online], 2023. https://www.semianalysis.com/p/ the-inference-cost-of-search-disruption . Kwon, W., Li, Z., Zhuang, S., Sheng, Y., Zheng, L., Yu, C.H.,Gonzalez,J.,Zhang,H.,andStoica,I. Efficient memorymanagementforlargelanguagemodelserving with pagedattention. In Proceedings of the 29th Sym- posiumonOperatingSystemsPrinciples ,pp.611–626, 2023. Langley,P.Craftingpapersonmachinelearning.InLangley, P.(ed.), Proceedingsofthe17thInternationalConference onMachineLearning(ICML2000) ,pp.1207–1216,Stan- ford,CA,2000.MorganKaufmann. Lefaudeux, B., Massa, F., Liskovich, D., Xiong, W., Caggiano, V., Naren, S., Xu, M., Hu, J., Tin- tore, M., Zhang, S., Labatut, P., and Haziza, D. xformers: A modular and hackable trans- former modelling library. https://github.com/ facebookresearch/xformers ,2022. Merity,S.,Xiong,C.,Bradbury,J.,andSocher,R. Pointer sentinelmixturemodels,2016. Nair,V.andHinton,G.E. Rectifiedlinearunitsimprove restrictedboltzmannmachines.In Proceedingsofthe27th internationalconferenceonmachinelearning(ICML-10) , pp.807–814,2010.

FlashDecoding++:FasterLargeLanguageModelInferencewithAsynchronization,FlatGEMMOptimization,andHeuristics Nerdynav. Up-to-datechatgptstatisticsandusernumbers Thirunavukarasu,A.J.,Ting,D.S.J.,Elangovan,K.,Gutier- [oct 2023]. [Online], 2023. https://nerdynav. rez, L., Tan, T. F., and Ting, D. S. W. Large language com/chatgpt-statistics . modelsinmedicine. Naturemedicine ,29(8):1930–1940, 2023. NVIDIA. Nvidia tensorrt: An sdk for high-performance deep learning inference. [Online]. https:// Touvron, H., Martin, L., Stone, K., Albert, P., Almahairi, developer.nvidia.com/tensorrt . A., Babaei, Y., Bashlykov, N., Batra, S., Bhargava, P., Bhosale,S.,etal. Llama2: Openfoundationandfine- NVIDIA. Cutlass: Cudatemplatesforlinearalgebrasub- tuned chat models. arXiv preprint arXiv:2307.09288 , routines. [Online],2017a. https://github.com/ 2023. NVIDIA/cutlass . Vaidya, N., Oh, F., and Comly, N. Optimizing inference NVIDIA. Fastertransformer: Abouttransformerrelatedop- onlargelanguagemodelswithnvidiatensorrt-llm,now timization,includingbert,gpt. [Online],2017b. https: publiclyavailable. [Online],2023. https://github. //github.com/NVIDIA/FasterTransformer . com/NVIDIA/TensorRT-LLM . NVIDIA. cublas: Basiclinearalgebraonnvidiagpus. [On- Vaswani,A.,Shazeer,N.,Parmar,N.,Uszkoreit,J.,Jones, line],2017c. https://developer.nvidia.com/ L., Gomez, A. N., Kaiser, Ł., and Polosukhin, I. At- cublas . tentionisallyouneed. Advancesinneuralinformation NVIDIA. Nvidia tensor core. [Online], 2023. https: processingsystems ,30,2017. //www.nvidia.com/en-us/data-center/ Wang,S. Fastgemv: High-speedgemvkernels. [Online], tensor-cores/ . 2023. https://github.com/wangsiping97/ OpenAI. Openai pricing. [Online], 2023. https:// FastGEMV . openai.com/pricing . Wolf,T.,Debut,L.,Sanh,V.,Chaumond,J.,Delangue,C., Moi, A., Cistac, P., Rault, T., Louf, R., Funtowicz, M., Pham, A., Yang, C., Sheng, S., Zhao, S., Lee, S., Jiang, Davison,J.,Shleifer,S.,vonPlaten,P.,Ma,C.,Jernite, B., Dong, F., Guan, X., and Ming, F. OpenLLM: Op- eratingLLMsinproduction,June2023. URL https: Y.,Plu,J.,Xu,C.,LeScao,T.,Gugger,S.,Drame,M., //github.com/bentoml/OpenLLM . Lhoest,Q.,andRush,A. Transformers: State-of-the-art naturallanguageprocessing. In Proceedingsofthe2020 Ramachandran,P.,Zoph,B.,andLe,Q.V. Searchingfor ConferenceonEmpiricalMethodsinNaturalLanguage activationfunctions. arXivpreprintarXiv:1710.05941 , Processing: SystemDemonstrations ,pp.38–45,Online, 2017. October2020.AssociationforComputationalLinguistics. doi: 10.18653/v1/2020.emnlp-demos.6. URL https: Rozi ere,B.,Gehring,J.,Gloeckle,F.,Sootla,S.,Gat,I.,Tan, ` //aclanthology.org/2020.emnlp-demos.6 . X.E.,Adi,Y.,Liu,J.,Sauvestre,R.,Remez,T.,Rapin,J., Kozhevnikov,A.,Evtimov,I.,Bitton,J.,Bhatt,M.,Ferrer, Xiao, G., Tian, Y., Chen, B., Han, S., and Lewis, M. Ef- C.C.,Grattafiori,A.,Xiong,W.,D efossez,A.,Copet,J., ´ ficientstreaminglanguagemodelswithattentionsinks. Azhar,F.,Touvron,H.,Martin,L.,Usunier,N.,Scialom, arXivpreprintarXiv:2309.17453 ,2023. T., and Synnaeve, G. Code llama: Open foundation Zellers,R.,Holtzman,A.,Bisk,Y.,Farhadi,A.,andChoi, modelsforcode. arXivpreprintarXiv:2308.12950 ,2023. Y. Hellaswag: Can a machine really finish your sen- Sakaguchi,K.,Bras,R.L.,Bhagavatula,C.,andChoi,Y. tence? In Proceedingsofthe57thAnnualMeetingofthe Winogrande: Anadversarialwinogradschemachallenge AssociationforComputationalLinguistics ,2019. atscale. arXivpreprintarXiv:1907.10641 ,2019. Zhang, S., Roller, S., Goyal, N., Artetxe, M., Chen, M., Sensetime. Openppl: A high-performance deep learn- Chen, S., Dewan, C., Diab, M., Li, X., Lin, X. V., Mi- ing inference platform. [Online], 2023a. https: haylov,T.,Ott,M.,Shleifer,S.,Shuster,K.,Simig,D., //openppl.ai/home . Koura, P. S., Sridhar, A., Wang, T., and Zettlemoyer, L. Opt: Openpre-trainedtransformerlanguagemodels, Sensetime. Alightandfastinferenceserviceforllm. [On- 2022. line], 2023b. https://github.com/ModelTC/ lightllm .

#### A D ETAILED LLM D ATAFLOW IN

#### Sheng,Y.,Zheng,L.,Yuan,B.,Li,Z.,Ryabinin,M.,Chen, FlashDecoding++ WITH K ERNEL F USION

B.,Liang,P.,Re,C.,Stoica,I.,andZhang,C. Flexgen: High-throughputgenerativeinferenceoflargelanguage FlashDecoding++ utilizesopen-sourcekernelsandfuses modelswithasinglegpu. 2023. kernelsinLLMs. Element-wisekernelslikeactivationand

FlashDecoding++:FasterLargeLanguageModelInferencewithAsynchronization,FlatGEMMOptimization,andHeuristics

V M E no EP t a e s e no G original m r o i t c e o c eh op s l u x a l u s op i t c l aud m r o V l aud Llama2 N S j o r R c a na m t a m tf m t a s na e j o i N S t od M E i M p V c r t V m o s m r t r p s e r M G s e r R Q VK Q O O R K K V U M L E G I S

m j o t a c no no + m V M l V fused r o r P EP eh i i t c l aud r o E U L t od aud M Llama2 N S V o R c a t ne e j o i N S G I S + i s E G M Q K + c VK tt a r p s e M l aud + e r + R O r R +

Figure18. ExampleofkernelfusionofLlama2dataflow. positionencodingarefusedwithlinearkernels. Weshow two types of AMD GPUs. However, the flat GEMM anexampleofkernelfusionofLlama2dataflowcompared optimization uses the Tensor Core, so we need different totheoriginaldataflowinFigure18. implementation approaches for the RX7900XTX and MI210. Given that MI210 has Matrix Cores, a hardware B I AMD structure similar to Tensor Cores for efficient matrix MPLEMENTATION ON computation,wemigrateCUDAcodeandadjustthewarp DuetothePyTorch’ssupportforAMDGPUs,wecanper- size to 64 to suit this GPU. RX7900XTX does not have formlargelanguagemodelinferenceonAMDGPUssimilar Matrix Cores, preventing direct code migration. To this towhatwedoonNVIDIAGPUs. Wehaveimplemented end, we use the WMMA compiler intrinsics, such as andvalidatedtheeffectivenessofourproposedmethodson builtin amdgcn wmma f16 16x16x16 f16 w32 , AMDGPUsusingAMDparallelprogramming.AMDparal- to develop flat GEMM kernels resulting in 20% speedup lelprogrammingsharesmanysimilaritieswithNVIDIApar- thanthetorch.matmulusedinPyTorchontheRX7900XTX. allelprogramming. Theirprogrammingmodelsaredivided into grid, block, warp, and thread. Similar to the CUDA runtimelibraryofNVIDIA,AMDhastheROCmruntime library. WecanuseHIPtodevelopkernelsforAMDGPUs. HIPhasAPIsthatcloselyresembleCUDAAPIs. Wecan easily port CUDA code developed for NVIDIA GPUs to HIPcodeforAMDGPUsbymodifyingtheAPInamesor usingtheHIPIFYtool(AMD,2023).However,architectural differencesbetweenGPUsmeanthatefficientkernelsde- velopedforNVIDIAGPUsmaynotnecessarilybeefficient onAMDGPUs,andinsomecases,theymaynotevenrun. Forexample,consumer-levelGPUliketheRX7900XTX, basedontheRDNA3architecture,lacksstructuressimilar to the Tensor Core and cannot efficiently perform matrix operationsusingWMMAinstructionsasCUDA.Incontrast, compute-levelGPUliketheMI210,basedontheCDNA2 architecture, has the Matrix Core but with a warp size of 64,unlikeNVIDIAGPUs. Thisnecessitatesoptimizations tailoredforeachoftheseGPUs. We employ different strategies for our implementations on these two types of AMD GPUs to accommodate their distinct characteristics compared to NVIDIA GPUs. Since our asynchronized softmax optimization for decode phase does not use the Tensor Core, we migrate CUDA codes to HIP and run them on these

