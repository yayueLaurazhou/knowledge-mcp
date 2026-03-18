# FlashDecoding++: Faster Large Language Model Inference with Asynchronization, Flat GEMM Optimization, and Heuristics

*Author: Ke Hong, Guohao Dai, Jiaming Xu, Qiuli Mao, Xiuhong Li, Jun Liu, Kangdi Chen, Yuhan Dong, Yu Wang*

---


<!-- page 1 -->


#### FD++:FLLMILASHECODINGASTERARGEANGUAGEODELNFERENCEWITH


#### A,FGEMMO,HSYNCHRONIZATIONLATPTIMIZATIONANDEURISTICS



*12*32*32124322KeHongGuohaoDaiJiamingXuQiuliMaoXiuhongLiJunLiuKangdiChen  

11YuhanDongYuWang  

ABSTRACT  

AstheLargeLanguageModel(LLM)becomesincreasinglyimportantinvariousdomains,theperformanceof  

LLMinferenceiscrucialtomassiveLLMapplications.However,thefollowingchallengesstillremainunsolved  

inacceleratingLLMinference:(1)Synchronizedpartialsoftmaxupdate.Thesoftmaxoperationrequiresa  

synchronizedupdateoperationamongeachpartialsoftmaxresult,leadingto∼20%overheadsfortheattention  

computationinLLMs.(2)Under-utilizedcomputationofflatGEMM.TheshapeofmatricesperformingGEMM  

inLLMinferenceisflat,leadingtounder-utilizedcomputationand50%performancelossafterpaddingzerosin  

previousdesigns(e.g.,cuBLAS,CUTLASS,etc.).(3)Performancelossduetostaticdataflow.Kernelperformance  

inLLMdependsonvariedinputdatafeatures,hardwareconfigurations,etc.Asingleandstaticdataflowmaylead  

toa50.25%performancelossforGEMMsofdifferentshapesinLLMinference.  

WepresentFlashDecoding++,afastLLMinferenceenginesupportingmainstreamLLMsandhardwareback-ends.  

Totackletheabovechallenges,FlashDecoding++creativelyproposes:(1)Asynchronizedsoftmaxwithunified  

maxvalue.FlashDecoding++introducesaunifiedmaxvaluetechniquefordifferentpartialsoftmaxcomputations  

toavoidsynchronization.Basedonthis,thefine-grainedpipeliningisproposed,leadingto1.18×and1.14×forthe  

prefillanddecodingstageinLLMinference,respectively.(2)FlatGEMMoptimizationwithdoublebuffering.  

FlashDecoding++pointsoutthatflatGEMMswithdifferentshapesfacevariedbottlenecks.Then,techniques  

likedoublebufferingareintroduced,resultinginupto52%speedupfortheflatGEMMoperation.(3)Heuristic  

dataflowwithhardwareresourceadaptation.FlashDecoding++heuristicallyoptimizesdataflowusingdifferent  

hardwareresource(e.g.,TensorCoreorCUDAcore)consideringinputdynamics.Thedesignleadstoupto  

29%speedupcomparedwiththestaticdataflow.DuetotheversatilityofoptimizationsinFlashDecoding++,  

FlashDecoding++××canachieveupto4.86and4.35speeduponbothNVIDIAandAMDGPUscompared  

FlashDecoding++×toHuggingFaceimplementations.alsoachievesanaveragespeedupof1.37comparedto  

state-of-the-artLLMinferenceenginesonmainstreamLLMs.  


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



1INTRODUCTION  

AstheLargeLanguageModel(LLM)achievedunprece-  

dentedsuccessinvariousdomains(Thirunavukarasuetal.,  

2023;Aniletal.,2023;Clusmannetal.,2023;Cuietal.,  

2023),theLLMinferenceworkloadisskyrocketing.For0  


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

example,OpenAIreportsthatGPT-4inferencewith8Kcon-  

textlengthcosts$0.03per1Kinputtokensand$0.06per  

1Koutputtokens(OpenAI,2023).Currently,OpenAIhas  

180.5millionusersandreceivesover10millionqueries  

perday(Nerdynav,2023).Consequently,thecosttoop-  

erateOpenAI’smodellikeChatGPTisapproximately$70  

Figure1.OverviewofcomparisonbetweenFlashDecoding++and  

*Equalcontribution1TsinghuaUniversity2Infinigence-AIstate-of-the-artdesigns.Theresultsinthefigurearereportedwith  

34ShanghaiJiaoTongUniversityPekingUniversity.Correspon-  

Llama2-7Bmodel(Touvronetal.,2023).Theleftiswithbatch  

denceto:GuohaoDai<daiguohao@sjtu.edu.cn>,YuWang  

size=1andinputlength=1K,andTensorRT-LLMandHugging<yuwang@tsinghua.edu.cn>.  

FacearetheSOTAbaselineforNVIDIA/AMDaccordingtoour  

thexperimentalresults.Therightshowsthecomprehensivecompari-Proceedingsofthe7MLSysConference,SantaClara,CA,USA,  

2024.Copyright2024bytheauthor(s).sonofbothfirsttokenlatencyandeachtokenlatency.  


<!-- page 2 -->

FlashDecoding++:FasterLargeLanguageModelInferencewithAsynchronization,FlatGEMMOptimization,andHeuristics  

(a)**Synchronized partial softmax update**(b)  

**partial attention**①partial softmax  


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

×**W(e.g., FlashAttention)**Attention summN-1  


| mul1 | max | exp |
| ---- | --- | --- |


| ul2 | Attention
N+1 |
| --- | ------------- |

synchronized updateemul& mul refer to s1 2×operation ②&④ in (a)a**Section 3**②**PAttention**④h**W**p××  


|     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- |
|     |     |     |     |     |     |
|     |     |     |     | P   |     |


|     |     |     |     |
| --- | --- | --- | --- |
|     |     |     |     |
| tt  | ent | io  | n   |


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

③**softmax**Attention Attention llmulexpmuli**ocean**N-112N+1ef**Pacific?**r×**WFFNFFN**unified max valuesumP**O12**asynchronized  

⑤⑥⑥**Asynchronized softmax with unified max value**×**W**  

**Under-utilized computation of flat GEMM**(c)  

padding zerosdirectly computing  

n①②③④⑤⑥Ao×Bor  


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

tizeroaflat-shapecomputation under-utilizationrGEMMe  

p×**Section 4**ABO  


|     |     |     |
| --- | --- | --- |
|     |     |     |
| K   | c   | ac  |
|     |     |     |


| A×B     | load A’’ | A’’×B     |        |
| ------- | -------- | --------- | ------ |
| load A’ | A’×B     | load A’’’ | A’’’×B |

load A**partial attention**①  

**(e.g., FlashDecoding)**double×**WK**buffering  

**Flat GEMM optimization with double buffering**e**he**  

sa×**Performance loss to static dataflow**(d)  

h②③**softmax**④p**W**××**Q** **PacificOcean**GEMM√ Flat GEMM× GEMV×e  


| GEMM           |
| -------------- |
| Flat GEMM
GEMV |


| static dataflow 1 |
| ----------------- |
| static dataflow 2 |

d×**WFFNFFNO12**o⑤⑥⑥GEMM× Flat GEMM√ GEMV√ce  


|     |     |     |
| --- | --- | --- |
|     |     |     |
|     |     |     |
| V   | c   | ac  |
|     |     |     |

D**WSection 5**×**V**  


| GEMM      |
| --------- |
| Flat GEMM |
| GEMV      |

**he**heuristic **autogressively**GEMM√ Flat GEMM√ GEMV√dataflow  

**Heuristic dataflow with hardware resource adaption**  

Figure2.OverviewofLargeLanguageModelinferencedataflow.FlashDecoding++proposesthreesolutionsforcorrespondingchallenges  

inLargeLanguageModelinference.(a)Thedataflowcomparisonbetweentheprefillphaseandthedecodephase.Theprefillphasemainly  

involvestheGEMMoperation,whilethedecodephasemainlyinvolvestheGEMV/FlatGEMMoperation.(b)FlashDecoding++proposes  

theasynchronizedsoftmaxwithunifiedmaxvaluetechnique,avoidingsynchronizedupdatetopreviouspartialattentionresults.(c)  

FlashDecoding++optimizesflatGEMMbyimprovingcomputationutilization.(d)FlashDecoding++heuristicallyoptimizesdataflow.  

millionperdayforthenecessarycomputinghardware(DY-Figure2(a)showsthemaindataflowoftheLLMinference  

LANPATEL,2023).Thus,optimizationsonLLMinferencewithonetransformerlayerforboththeprefillphaseand  

performancewillhaveahugeimpactconsideringmassivethedecodephase.Atransformerlayercanbedividedinto  

LLMinferencescenarios.Manyrecentworkshavepro-linearGEMM(GeneralMatrixMultiplication)operations  

posedtechniquestoaccelerateLLMinferencetasks,includ-(e.g.,K,Q,V,Oweightprojectionandthefeedforward)  

ingDeepSpeed(Aminabadietal.,2022),FlexGen(Shengandtheattention/softmaxcomputation.Fortheattention  

etal.,2023),vLLM(Kwonetal.,2023),OpenPPL(Sense-computation,asoftmaxoperationisadoptedforarowin  

time,2023a),FlashDecoding(Daoetal.,2023),TensorRT-theattentionmatrix.Toimprovetheparallelism,previous  

LLM(Vaidyaetal.,2023),andetc(Sensetime,2023b;TGI,designs(Daoetal.,2022;2023)dividetheattentionma-  

2023;mlc,2023;Sensetime,2023a).tricesintosmallertilesandrowsarealsosplittocompute  

partialsoftmaxresults.Asynchronizedsoftmaxoperation  

TheLLMinferencetaskgeneratestokens(e.g.,words)from  

isadoptedtoupdatepreviouspartialsoftmaxresultswhen  

theinputsequenceautoregressively,andcanbeorganized  

anewpartialsoftmaxresultiscalculated.Suchasynchro-  

intotwotypicalphases:theprefillphaseandthedecode  

nizedpartialsoftmaxupdateaccountsfor18.8%forthe  

phase.Theprefillphasegeneratesthefirsttokenbyprocess-  

attentioncomputationofLlama2-7Binferenceaccording  

ingtheinputprompt,andpreviousresearch(e.g.,FlashAt-  

toourprofilingonNVIDIATeslaA100GPUwith1024  

tention(Daoetal.,2022;Dao,2023))optimizeslatencyfor  

inputlength,resultinginthefirstchallengeforaccelerat-  

thisphase.Thedecodephasegeneratesthefollowingto-  

ingLLMinference.Secondly,thecomputationresources  

kenssequentially,andmanyworks(Aminabadietal.,2022;  

isunder-utilizedfortheflatGEMMoperationduring  

Shengetal.,2023;Kwonetal.,2023;Sensetime,2023b;  

thedecodephase.Becausethedecodephasesequentially  

Daoetal.,2023;Vaidyaetal.,2023;Phametal.,2023)fo-  

generatestokens,thelinearGEMMoperationtendstobe  

cusonimprovingthethroughputofgeneratingtokens(i.e.,  

flat-shape(eventurningintotheGEMV(GeneralMatrix-  

reducinglatencyofeachtoken).Theprefillphasedominates  

VectorMultiplication)operationwhenthebatchsizeis1).  

totaltimeforscenariosoflong-sequenceinputorgenerat-  

Forthesmallbatchsize(e.g.,8),previousdesigns(NVIDIA,  

ingshortoutputs(Daietal.,2019;Dongetal.),whilethe  

2017c;a)padthematrixwithzerostoperformGEMMsof  

decodephaseconstitutesasignificantportionofthetime  

e.g.,largersizes(64),leadingtoover50%computation  

whenprocessinglongoutputsequences(Xiaoetal.,2023).  


<!-- page 3 -->

FlashDecoding++:FasterLargeLanguageModelInferencewithAsynchronization,FlatGEMMOptimization,andHeuristics  

under-utilization.Thirdly,theperformanceofLLMin-2BACKGROUND  

ferencesuffersfromthestaticdataflowconsideringinput  

2.1LLMInferenceDataflowOverviewdynamicsandhardwareconfiguration.Forexample,the  

smallbatchsizemakesthedecodephaseofLLMinferenceThetaskofLLMinferenceistogeneratetokensfromthe  

memory-boundedandthelargebatchsizemakesitcompute-inputsequence,whichcanbeusedtocompleteasentence  

bounded.Asingleandstaticdataflowmayleadto50.25%oransweraquestion.AnoverviewoftheLLMinference  

performancelossforGEMMsofdifferentshapesinLLMdataflowisshowninFigure2(a).Aswecansee,theLLM  

inference.inferencedataflowcanbeorganizedintotwotypicalphases  

withsimilaroperations:oneprefillphaseandseveraldecodeTotacklethesechallengesandenableafasterLargeLan-  

phases.Theprefillphase“understands”theinputsequenceguageModel(LLM)inference,wepresentFlashDecod-  

(i.e.,“Whatisthelargestocean?”).Eachtoken(wesetoneing++inthispaper.FlashDecoding++creativelyproposes  

wordasatokeninFigure2(a))isencodedasanembeddingthefollowingcontributions:  

vector,andtheinputsequenceisorganizedintoamatrix.  

Themainoutputoftheprefillphaseisanewtoken,whichis  

•Asynchronizedsoftmaxwithunifiedmaxvalue.  

predictedtobethenexttokenaftertheinputsequence(i.e.,FlashDecoding++leveragesaunifiedmaxvaluefor  

decode“Pacific”inthisfigure).Thephase“generates”the  

differentpartialsoftmaxcomputations.Eachpartial  

i.e.,outputsequence(“Pacific”,“Ocean”,etc.)Theoutput  

softmaxresultcanbeprocessedindividuallywithout  

prefilldecodetokenofthephaseistakenastheinputofthesynchronizedupdate.Suchatechniqueleadsto1.18×  

decodephase.Thephaseisexecutedautogressively,andand1.14×speedupforattentioncomputationinthe  

eachoutputtokenisusedastheinputtokenforthenextTheprefillstageanddecodingstage,respectively.  

decode(e.g.,“Ocean”isfurtherusedastheinput).  

•FlatGEMMoptimizationwithdoublebuffering.  

FlashDecoding++onlypadsthematrixsizeto8rather2.2OperationsinLLMInference  

than64inpreviousdesignsforflat-shapedGEMMto  

ThemainoperationsinLLMinferencearedepictedasoper-improvecomputationutilization.Wepointoutthatflat  

①⑥ationtoinFigure2(a),includingthelinearprojectionGEMMswithdifferentshapesfacevariedbottlenecks,  

①⑤②③④(and),theattention(,,and),andthefeedforwardandfurtherimprovethekernelperformancebyupto  

⑥network().Forsimplicity,operationslikepositionembed-52%withtechniqueslikedoublebuffering.  

ding(Vaswanietal.,2017),non-linearactivation(Nair&  

•Heuristicdataflowwithhardwareresourceadap-Hinton,2010;Ramachandranetal.,2017),mask(Vaswani  

tion.FlashDecoding++takesbothinputdynamicsetal.,2017),andothersarenotshowninthefigure.Opera-  

andhardwareconfigurationsintoconsiderationandtionsintheprefillphaseandthedecodephasearedifferent  

dynamicallyapplieskerneloptimizationfortheLLMintheshapeofdata.Becauseonlyonetoken(batchsize=1)  

inferencedataflow.Suchatechniqueleadstouptoorfewtokens(batchsize>1)areprocessedatonetime,in-  

29%speedup.putmatricesinthedecodephaseareflat-shapematrices  

orevenvectors.  

Becauseoftheversatilityofoptimizations,theeffective-  

LinearProjection.Thelinearprojectionperformsasthe  

nessofFlashDecoding++canbeprovedonbothNVIDIA  

fullyconnectedlayer,multiplyingtheinputwithweightma-  

andAMDGPUs.FlashDecoding++achievesupto4.86×  

trices(i.e.,W,W,W,W,calledK,Q,VprojectionKQVOand4.35×speeduponbothNVIDIAandAMDGPUscom-  

andOprojection).Fortheprefillphase,theK,Q,Vprojec-  

paredwithHuggingFaceimplementations,respectively.  

tiongeneratesmatricesK,Q,V.Forthedecodephase,the  

OurextensiveresultsshowthatFlashDecoding++achieves  

K,Q,Vprojectiongeneratesthreecorrespondingvectors  

anaverageof1.37×speedupcomparedwithFlashDecod-  

andconcatenatedwithKandV(i.e.,KVcache,yellowand  

ing(Daoetal.,2023),astate-of-the-artLLMinference  

lightblueinFigure2(a))intheprefillphase.  

engineonvariousLLMs(e.g.,Llama2,ChatGLM2,etc.).  

softmax(Q×KT)×V(1)  

Therestofthispaperisorganizedasfollows.Section2in-  

troducespreliminariesofLLMsandrelatedworksonLLMAttention.Theattentionoperationismainlydividedinto  

②④inferenceacceleration.Ourthreetechniques,theasynchro-threeoperations(toQ×K,softmax,Attention×  

Tnizedsoftmaxwithunifiedmaxvalue,theflatGEMMopti-V),asshowninEq.(1).ForP=Q×K,thesoftmax  

mizationwithdoublebuffering,andtheheuristicdataflowoperationisperformedforeachrowoftheresultmatrixofP.  

withhardwareresourceadaptionaredetailedinSection3,4,ThedetailedsoftmaxcomputationisshowninFigure3(a).  

and5,respectively.Section6presentstheevaluationre-Themaximumvaluem(x)isfirstcalculated.Theexponent  

m(x)sults.RelatedworksonLLMinferenceareintroducedinofeachelementdividedbye,f(x),isthenprocessed.  

Section7,andSection8concludesthepaper.Theseexponentsarenormalizedtothesummationofall  


<!-- page 4 -->

FlashDecoding++:FasterLargeLanguageModelInferencewithAsynchronization,FlatGEMMOptimization,andHeuristics  

𝑠𝑜𝑓𝑡𝑚𝑎𝑥𝑥𝑠𝑜𝑓𝑡𝑚𝑎𝑥𝑥′𝑠𝑜𝑓𝑡𝑚𝑎𝑥𝑥′′𝑠𝑜𝑓𝑡𝑚𝑎𝑥𝑥′𝑠𝑜𝑓𝑡𝑚𝑎𝑥𝑥′′  


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



𝑚𝑥=𝑚𝑎𝑥𝑥Calcutate𝑚𝑥′,𝑓𝑥′,𝑙𝑥′,𝑠𝑜𝑓𝑡𝑚𝑎𝑥𝑥′𝑚𝑥′=𝑚𝑥′′=aunifiedmaxvalue𝜙!  

Calcutate𝑚𝑥′′,𝑓𝑥′′,𝑙𝑥′′,𝑠𝑜𝑓𝑡𝑚𝑎𝑥𝑥′′Calcutate𝑓𝑥′,𝑙𝑥′𝑓𝑥=𝑒"'#$",𝑒"(#$",…,𝑒")#$"  

Update𝑠𝑜𝑓𝑡𝑚𝑎𝑥𝑥′with𝑚𝑥′,𝑓𝑥′,𝑙𝑥′,𝑚𝑥′′,𝑓𝑥′′,𝑙𝑥′′Calcutate𝑓𝑥′′,𝑙𝑥%′𝑙𝑥=+𝑓𝑥!  

Update𝑠𝑜𝑓𝑡𝑚𝑎𝑥𝑥′′with𝑚𝑥′,𝑓𝑥′,𝑙𝑥′,𝑚𝑥′′,𝑓𝑥′′,𝑙𝑥′′……!𝑓𝑥  

𝑠𝑜𝑓𝑡𝑚𝑎𝑥𝑥=𝑙𝑥ProcessnextpartialvectorCalcutate𝑠𝑜𝑓𝑡𝑚𝑎𝑥𝑥  

Highparallelism×Highparallelism√Highparallelism√  

Lowmemory×Lowmemory√Lowmemory√  

Synchronization-free√Synchronization-free×Synchronization-free√  

(a)Originalsoftmax(b)Partialsoftmax(c)Partialsoftmaxwithunifiedmaxvalue  

Figure3.Comparisonofdifferentsoftmaxcomputationschemes.(a)Softmaxcomputationforthewholevector.(b)Computingpartial  

softmaxforeachpartialvector,andasynchronizedupdateoperationisrequiredforallpartialsoftmaxresults.(c)Computingpartial  

softmaxusingaunifiedmaxvalue,andeachpartialvectorisprocessedindividuallywithoutsynchronizedupdate.  

3ASexponents(i.e.,l(x))togetthesoftmaxresult.SYNCHRONIZEDOFTMAXWITH  

UMVNIFIEDAXIMUMALUEFeedforwardNetwork.Thefeedforwardnetworkprimar-  

⑥ilycomprisestwofullyconnectedlayers.Thefirstone(Motivation.Thepartialsoftmaxoperationrequiressynchro-  

FFN)expandsthefeaturedimensionstoenhancetherep-1nizationamongdifferentpartialvectors,leadingto∼20%  

⑥resentationalcapacity.Thesecondone(FFN)restores2overheadsoftheattentionoperation.AsisshowninFig-  

thefeaturedimensionsandservesastheoutputlayer.ure2(b),thesynchronizationisrequiredafterthemaximum  

valueofthepartialvectoriscalculated.Themaximumvalue  

2.3AttentionOptimizationisusedtoupdatepreviouspartialsoftmax(i.e.,recompute  

previousattention)results.Thus,toreducesynchroniza-ThesoftmaxoperationshowninFigure3(a)requiresall  

tionoverheads,thekeyproblemtobesolvedishowtoglobaldatatobecalculatedandstoredbeforeitcanpro-  

computeeachpartialsoftmaxresultwithoutrequiring  

ceed.Thisresultsinhighmemoryconsumptionandlow  

resultsfromotherpartialsoftmaxcomputation.  

parallelism.Latterworksproposethepartialsoftmaxtech-  

niquetoreducememoryconsumption(Daoetal.,2022;Challenge.Thereasonthatsynchronizationisrequired  

Dao,2023)orimproveparallelism(Daoetal.,2023).Fig-liesinthatthemaximumvalueofeachpartialvectoris  

ure3(b)showsthediagramofthepartialsoftmaxoperation.different.Themaximumvalueisusedtoavoidoverflowof  

Themainideaistodividethevectorxintopartialvectorstheexponentoperation(f(x)inFigure3(a)),andexponents  

(i.e,x′andx′′).Thepartialsoftmaxresultsofx′andx′′aresummed(l(x)inFigure3(a))asthedenominatorofthe  

arecalculatedseparatelyaccordingtoFigure3(a),andthensoftmaxoperation.Suchanon-linearoperationoneach  

synchronouslyupdatedbyeachother.Thedetailedcompu-partialmaximumvaluemakesthesynchronizationamong  

tationofthissynchronizedupdateisshowninEquation(2).eachpartialsoftmaxcomputationunavoidable.  

Withtheimplementationofpartialsoftmax,wecanachieve  

AnalysisandInsights.Accordingtotheformulaofsoftmax  

efficientparallelismofcomputationwhilereducingmemory  

computation,themaximumvalueisusedasthescalingfac-  

costforattentioncomputation.  

torforboththenumeratorandthedenominator(i.e.,f(x)  

m(x)=max(m(x′),m(x′′))andl(x)inFigure3(a)).Ourkeyinsightis,thescaling  

′m(x′)−m(x)′factorcanbeanarbitrarynumberratherthanusingthef(x)=ef(x)  

maximumvaluemathematically,showninEquation(3).  

′′m(x′′)−m(x)′′(2)f(x)=ef(x)Whenwesetϕ=0,itbecomestheoriginalsoftmaxcom-  

′′′l(x)=f(x)+f(x)putation(Bridle,1989).  

′′′′′′softmax([x,x])=[f(x),f(x)]÷l(x)  

x−m(x)x−m(x)[e1,...,ed]  

softmax(x)=However,sincethepartialsoftmaxneedstobeupdated**(cid:80)**ex−m(x)iiaccordingtootherpartialsoftmaxresults,itunavoidably(3)  

x−ϕx−ϕ[e1,...,ed]introducesdatasynchronizationoperations.AccordingtoR=,∀ϕ∈**(cid:80)**ex−ϕourprofilingresult,suchasynchronizedupdateoperationii  

leadsto18.8%overheadsintheattentioncomputationfor  

Llama2-7BinferenceonNVIDIATeslaA100GPUwithHowever,thescalingfactorcannotbeanarbitrarynumber  

1024inputlength.consideringtheoverflowingoftheexponentcomputation.  


<!-- page 5 -->

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

xv  


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



getx,xgetx,x1234  

fromQ,KfromQ,K  

calculatenumerator+=calculatenumerator+=numerator!  

e4-6,e5-6e4-6·v+e5-6·ve6-6,e7-6e4-6·v+e5-6·vdenominator1212  

denominator+=denominator+=  

e4-6+e5-6e4-6+e5-6  

(a)Calculatesoftmax(x)×vT  

yv  


| y=3
1 | y=6
2 | y=10
3 | y=7
4 |
| ----- | ----- | ------ | ----- |

vvvv1234  

gety,ygety,y1234**recomputationprocess**fromQ,KfromQ,K  

Figure4.Thestatisticaldistributionofx(elementsintheinputcalculatenumerator+=calculatecomputecomputeupdateupdateie3-6,e6-6e3-6·v+e6-6·ve10-6,e7-6softmaxsoftmaxsoftmaxsoftmaxvectorsofsoftmax)intypicalLLMswithdifferentinputs.121212  

denominator+=**10-6>3,overflow!**e3-6+e6-6x−ϕForthecasewherex≫ϕ,eioverflowsandcannotbei(b)Calculatesoftmax(y)×vT  

representedusingafix-widthfloatingpointnumber(e.g.,  

Figure5.Exampleofasynchronizedpartialsoftmaxcomputation.float32forexponentresultsincurrentLLMengines).  

(a)Eachpartialsoftmaxresultisprocessindividuallywithoutthex−ϕForanothercasewherex≪ϕ,ei→0,leadingtoisynchronizedupdate.(b)Therecomputationprocessforallparital  

precisionloss.Thus,aproperscalingfactorϕshouldbe  

softmaxcomputationisrequiredwhenoverflowhappens.  

carefullyselectedtoavoidthetwocasesabove.Figure4  

x−ϕ≥bx,theasynchronizedpartialsoftmaxcomputationshowsthestatisticaldistributionof(elementsinthein-ii  

xxisterminatedforthevectorwherebelongsto.Theputvectorsofsoftmax)intypicalLLMswithdifferentin-i  

>99.99%xsoftmaxisthenrecomputedusingthesynchronizedpartialputs(Merityetal.,2016).Ourkeyinsightis,i  

arewithinacertainrangesoftmaxscheme(usedinFlashAttention(Daoetal.,2022;.Specifically,forLlama2-7B,  

−16.8<x<6.5>99.99%xDao,2023)andFlashDecoding(Daoetal.,2023))shownwehavefor.Becauseii  

inFigure3(b).Sucharecomputationschemeavoidsover-eb−aandea−bcanberepresentedbyafloat32format,  

flowwhileintroducingnegligibleoverheadsbasedonthewecansetϕ=ainEquation(3).ForOPT-6.7B,wedo  

statisticaldatashowninFigure4.notapplythetechniqueinthissectionbecauseofthelarge  

rangeinFigure4.Example.Figure5showsanexampleoftheasynchronized  

softmaxscheme.Weseta=−3,b=3,ϕ=6.TwoApproach:Asynchronization.Basedontheinsightsabove,  

vectorsxandyarecalculatedfromQ×KTinEquation(1),eachpartialsoftmaxcomputationsharesaunifiedmaxi-  

andaredividedinto2partialvectors.Weomittheprocessmumvalue,ϕ.Afterthesoftmaxoperation,aninnerprod-  

fromQ×KTtothesepartialvectors.Foreachx,weuctoperationisexecutedbetweenthesoftmaxresultandi  

havea<x−ϕ<b,weprocessex−ϕ·v+ex−ϕ·vacolumnofV(i.e.,v).Assumethattheinputvectorx12i12  

andex−ϕ+ex−ϕforthefirstpartialvectorofxusingcanbedividedintoppartialvectors,x=[x(1),...,x(p)]12  

twoasynchronizedthreads.Then,eachthreadmovestothe(v=[v(1),...,v(p)]correspondingly),wehave:  

nextpartialvectorforthecorrespondingcomputation(i.e.,  

ex−ϕ·v+ex−ϕ·vandex−ϕ+ex−ϕ).Twothreads**(cid:80)**3434  

ex−ϕ·v34iiiaresynchronizedwhenallpartialvectorsareprocessed,and⟨softmax(x),v⟩=**(cid:80)**ex−ϕiperformthedivisionoperationinEquation(4).Fory,thei  

(4)**(cid:80)**p**(cid:80)**d/p(j)(j)firstpartialvectorisprocessedsimilarly.However,wefindex−ϕ·vij=1i=1i=thaty−ϕ>b,thentwothreadsareterminatedandthe3**(cid:80)**p**(cid:80)**d/p(j)ex−ϕifirstthreadrecomputesallpartialvectorsaccordingtothej=1i=1  

synchronizedpartialsoftmaxschemeinFigure3(b).  

Theinneraccumulationinboththenumeratorandthede-  

(j)(j)nominatoronlytakethepartialvectorsxandvasinput,  

4FGEMMOLATPTIMIZATIONWITHthustheycanbeprocessedasynchronouslyandindividu-  

DOUBLEBUFFERINGally.Theouteraccumulationisonlyprocessedafterall  

partialvectorsareprocessed.AswecanseeinFigure3(c),  

Motivation.Theprocessofthedecodephaseismainly(j)eachf(x)iscalculatedindividually,andsoftmax(x)is  

composedofGEMV(batchsize=1)orflatGEMM  

(j)calculatedafterallxiscalculated.  

(batchsize>1)operation.Withoutlossofgeneral-  

Approach:Recomputation.Withoutlossofgenerality,weity,GEMV/GEMMoperationscanberepresentedusing  

assumea<x−ϕ<bforeachxtoensureprecisionM,N,K,wherethesizesoftwomultipliedmatricesareii  

andavoidoverflow.Then,thepartialsoftmaxoperationM×KandK×N.Tilingisacommontechniquefor  

isprocessedindividually.However,whenx−ϕ≤aorcomputingGEMMsonGPUs.Theoriginalmatricesarei  


<!-- page 6 -->

FlashDecoding++:FasterLargeLanguageModelInferencewithAsynchronization,FlatGEMMOptimization,andHeuristics  

BBBC=A·B+A·B+A·B+…NNN1112233  

C=A·B’+A·B’+A·B’+…32641282565123264128256512B2112233K  


| B1  | B’1 |     |     |
| --- | --- | --- | --- |
| B2  | B’2 |     |     |
| B3  | B’3 |     |     |
| …   | …   |     |     |
|     |     |     | B   |
| C1  | C2  |     | C   |

10241024 in shared memory for loading  


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

20482048 in shared memory for computing  


| A1B’1 | idle |
| ----- | ---- |

40964096  

idle81928192  

1638416384NNABABAB’AB’327683276811221122  

6553665536  

ABABAB’AB’13107213107233223322  

e262144262144…in…elMmTiK=4096K=12288GPU BlockGPU Block12KN**√**B with the best flat GEMM performance for a certain NNFigure7.DoublebufferingforflatGEMMwhenN−dimensionis  


| A1  | A2  | A3  | …   |     | A   |
| --- | --- | --- | --- | --- | --- |

Figure6.NormalizedflatGEMMperformanceunderdifferent  

large.TheM−dimensionispaddedto8andnottiled.  

N−dimensionsizesandN−dimensiontilingsizes.Weset  

M=8andexecuteGEMMontheNVIDIATeslaA100GPU.  

exposingacontradictiononimprovingtheperformanceof  

tiledintomultiplesub-matrices,andthendistributedacrossGEMVorflatGEMM.Wedepictthenormalizedperfor-  

differentcomputingunitstoenableparallelprocessing.Pre-manceoftheflatGEMMinFigure6withdifferentNand  

viousLLMinferenceenginesutilizeTensorCoretoacceler-B.Ourkeyinsightis,forthesmallerN,theflatGEMMN  

atetheseoperationsusinglibrarieslikecuBLAS(NVIDIA,isparallelism-bounded.Thereare108StreamingMulti-  

2017c)andCUTLASS(NVIDIA,2017a).AlthoughmodernNprocessors(SMs)intheNVIDIATeslaA100.tends  

BNTensorCorearchitectures(NVIDIA,2023)processGEMMe.g.,tobeaconstant(128or256),whichisrelatedtothe  

withM=8,theselibrariesusuallytiletheM−dimensionhardwareparallelism(numberofSMs).Anotherkeyinsight  

to64tohidememorylatency.However,forGEMVorflatNis,forthelarger,theflatGEMMbecomesmemory-  

GEMMoperationsinthedecodephase,weusuallyhavebounded.Theperformanceofthesecasescanbeimproved  

M≪64andtheM−dimensionispaddedto64withze-byhidingmemoryaccesslatency.  

ros.Thepaddingleadstounder-utilizedcomputation,and  

Approach:DoubleBuffering.Inordertohidememory  

thekeyproblemistoprocessGEMVorflatGEMM  

accesslatency,weintroducethedoublebufferingtechnique.  

operationswithsmallertiles(i.e.,paddingto8corre-  

fortheflatGEMMoperation.Weallocatetwoseparate  

spondingtomodernTensorCorearchitectures)inthe  

buffersinthesharedmemory.Thetileinonebufferper-  

M−dimension.  

formstheGEMMoperation,whileanotherbufferloadsa  

Challenge.ProcessingGEMVorflatGEMMoperationsnewtileforthenextGEMMoperation.Thus,thecomputa-  

isnon-trivialwhentheM−dimensionispaddedto8.Thetionandthememoryaccessareoverlapped.Weapplysuch  

tilingtechniqueinmodernlibrarieslikecuBLAS(NVIDIA,atechniquewhenNislargeinourpractice.  

2017c)andCUTLASS(NVIDIA,2017a)canonlybeap-  

Example.Figure7showstheexampleofourflatGEMM  

pliedtotheN−dimensionandtheK−dimension.Tiles  

optimizationwithdoublebuffering.ForM<8,theontheK−dimensionareprocessedsequentiallyinaGPU  

M−dimensionisfirstpaddedto8consideringmodernTen-blocktoavoidatomicoperationsduringreduction.Tiling  

sorCorearchitectures.WorkloadsintheK−dimensionontheN−dimensionaffectsbothparallelismandcompu-  

areprocessedwithinoneGPUblock(e.g.,A,A,A,...),tation/memoryratio,whicharebothimportantforGEMV123  

whileworkloadsintheN−dimensionareprocessedinpar-andflatGEMMacceleration.  

allelusingdifferentGPUblocks(e.g.,C,C,...).Wetake12  

AnalysisandInsights.AssumethattilingsizesoftheGPUBlockasanexample,thefirsttileforeachmatrix1  

N−dimensionandtheK−dimensionareBandB,re-intheK−dimension(i.e.,AandB)isloadedtotheleftNK11  

spectively.ThecomputationofeachGEMMtileis2×M×bufferinthesharedmemory.Then,theGEMMoperationis  

N×KB×BwithtotalB=GEMMtiles.ThetotalperformedbetweenAandB.Consequently,AandBNK1122BN×BKmemoryaccessis(M×B+B×B)×B+M×N.areloadedtotherightbufferinthesharedmemory.Thefol-KNK  

Thus,thecomputation/memoryratiois:lowingtilesareprocessedsimilarlyaccordingtothedouble  

bufferingscheme.  

2×M×B×B×BNK  

(M×B+B×B)×B+M×NKNK  

(5)5HDEURISTICATAFLOWWITH2×M×K  

=HARDWARERESOURCEADAPTIONK+M×K+M  

BN  

Motivation.AlthoughFlashDecoding++optimizesthe  

NOntheotherhand,theparallelismis.Thus,thecompu-flatGEMMoperationinSection4,itdoesnotcoverall  

BNtation/memoryratioshowsapositivecorrelationwithBoperations(evenonlyforGEMMs)intheLLMinference.N  

whiletheparallelismshowsanegativecorrelationwithB,AsmentionedinFigure2(a),theshapesofGEMMsindif-N  


<!-- page 7 -->

FlashDecoding++:FasterLargeLanguageModelInferencewithAsynchronization,FlatGEMMOptimization,andHeuristics  

**For a certain LLM, traverse four [N, K] selections**  


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

M=1  

Impl.B >  

Impl.A?  

Find M++M1  

Impl.C >  

Impl.B?  

Find M++**K, Q, VOFFNFFN**M**12**2**projectionprojection**  

ImplA = FastGEMV  

ImplB = our flat GEMMEnd[N, K] =[N, K] =[N, K] =[N, K] =  

ImplC = CUTLASS[12288, 4096][4096, 4096][11008, 4096][4096, 11008]  

(a) Different shapes of GEMMs in LLM(b) Decision flow (c) Example of heuristic dataflow with hardware resource adaption  

Figure8.HeuristicdataflowwithhardwareresourceadaptioninFlashDecoding++.(a)Onlyfour[N,K]shapesexistforacertainLLM.  

(b)Thedecisionflow.Wetraverseall[N,K]selectionsandprofiletheperformanceofthreerepresentativeimplementations.Mis  

increasedtofindtwoinflectionpointsforruntimeheuristicdataflow.(c)FlashDecoding++heuristicallyutilizesTensorCore/CUDACore  

withthecorrespondingGEMV/GEMMimplementationbyreferringtoalookuptable.  

ferentoperationsandtwophasesvary.Thus,theGEMMimpactsthekernelperformance.Alltheseinfluentialfactors  

workloadintheLLMinferencecanbeGEMV(batchsize=1buildalargesearchspace,makingitnon-trivialtogenerate  

forthedecodephase),flatGEMM(smallbatchsizeforaneffectivemappingbetweenthelinearworkloadandthe  

thedecodephaseandshortsequencelengthfortheprefillcorrespondingoptimalimplementation.  

phase)andconventionalGEMM(largebatchsizeorlong  

AnalysisandInsights.Althoughallinfluentialfactorsform  

sequencelengthfortheprefillphase).Inordertoleverage  

alargesearchspace,thehomogeneityofdifferentlayers  

thepowerfulcomputationalabilityofTensorCore,cur-  

inLLMsignificantlyreducesthesearchspaceforoperator  

rentframeworkslikeFasterTransformer(NVIDIA,2017b)  

optimization.Figure2(a)showsfourlinearGEMV/GEMM  

andDeepSpeed(Aminabadietal.,2022)tendtoutilizethe  

operationsintheprefillphaseandthedecodephase,i.e.,  

highlyoptimizedGEMMimplementationfromcuBLAS  

K,Q,Vprojection,Oprojection,andtwofeedforwardop-  

(NVIDIA,2017c)todealwithdifferentworkloads.How-  

erations.EachGEMV/GEMMoperationcanbecanbe  

ever,theTensorCoreimplementationfailswiththeGEMV  

abstractedasamultiplicationbetweenan(M×K)-shaped  

workload.TheGEMVworkloadcanbeoptimizedbyutiliz-  

matrixanda(K×N)-shapedmatrix.Ourkeyinsightis,  

ingCUDACoreinpreviousdesignslikeFastGEMV(Wang,  

thereareonlyfour[K,N]shapesforacertainLLM.  

2023).ForaLlama2-7Blinearlayerinthedecodephase,the  

Moreover,Misonlyrelatedtotheinputsequencelength  

TensorCoreimplementationfromcuBLASonlyachieves  

andthebatchsizefortheprefillphase,andthebatchsize  

82.15%oftheperformanceofCUDACoreimplementation  

forthedecodephase.Figure8(a)showslimitedshapesof  

usingFastGEMVonanNVIDIAA100GPU.Ontheother  

GEMV/GEMMoperationsintheLLMinference.  

hand,usingCUDACoretodotheprojectiononabatch-  

size=4decodinginputonlyachieves49.75%performanceApproach:Decisionflowforinflectionpoints.Because  

comparedwiththeTensorCoreimplementation.Thus,inonlyfour[K,N]shapesexistforacertainLLM,weuse  

ordertoapproachtheoptimalcomputationperformance,athreetypesofimplementationsforGEMV/GEMMopera-  

heuristicdataflowissupposedtobeexploitedfordiffer-tionswhenMvaries:FastGEMVfortheGEMVandflat  

entworkloads.GEMMoperations(ImplA),ourflatGEMMoptimization  

inSection4(ImplB),andtheCUTLASS(NVIDIA,2017a)  

Challenge.Althoughaheuristicdataflowpotentiallyexists  

librariesoptimizedfortheconventionalGEMM(ImplC).  

intheimplementationofdifferentlinearworkloads,itis  

Thus,itisimportanttodecidewhetherapplyingImplAor  

challengingtobuildthemappingfromacertainworkload  

ImplBforasmallM,andImplBorImplCforalargeM.  

toanoptimalimplementation.InthescenarioofLLMinfer-  

Figure8(b)showsthedecisionflow.FlashDecoding++  

ence,therearevariousfactorsthatinfluencetheimplemen-  

profilestheperformanceofImplAandImplBforacertain  

tationperformanceoflinearworkloads:(a)Inputdynamics.  

M,andincreasesMtofindaninflectionpointMwhere1Thevarietyofthebatchsizeandtheinputsequencelength  

theperformanceofImplBisbetterthanImplA.Anotherin-  

bringsdynamicworkloads.(b)Modeldiversity.Thelinear  

flectionpointMisfoundsimilarlywheretheperformance2workloadvarieswithdifferentmodelstructuresandsizes.  

ofImplCisbetterthanImplB.Notethateach[N,K]gets  

(c)GPUcapacities.Therelativeperformancebetweenim-  

itsindividualMandM.12plementationschangeswithGPUcharacteristics,suchas  

memorybandwidth,cachesize,andcomputationalability.Approach:Heuristicdataflow.FortheruntimeLLM  

(d)Engineeringeffects.Theengineeringeffortalsohighlyinference,FlashDecoding++adoptsImplAusingCUDA  


<!-- page 8 -->


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

s  


|     |     |     |                 |     |     |         |     |     |
| --- | --- | --- | --------------- | --- | --- | ------- | --- | --- |
|     |     |     | 128 1k 2k 4k 12 |     |     | 8 1k 2k |     |     |
|     |     |     |                 |     |     |         |     |     |
|     |     |     | 128 1k 2k 4k 12 |     |     | 8 1k 2k |     |     |
|     |     |     |                 |     |     |         |     |     |
|     |     |     |                 |     |     |         |     |     |



Fe  

e  

)  



CorewhenM<M,andImplB/ImplCusingTensorCoreWeevaluatetheperformanceofFlashDecoding++and1  

whenM≤M<M/M≤M.NotethatthedecisionotherLLMenginesonbothNVIDIAandAMDplatforms122  

flowareexecutedoffline,itdoesnotaffecttheperformancetomakeacomprehensivecomparison.Wechoosetwodif-  

ofruntimeLLMinference.ferentGPUsforeachplatform:TeslaA100andRTX3090  

forNVIDIA,MI210andRX7900XTXforAMD.Weshow  

Example.Figure8(c)showsanexampleofapplyingthe  

thedetailedconfigurationinTable1.  

heuristicdataflowfortheLlama2-7Bmodel.Four[N,K]  

shapesare[12288,4096]forK,Q,Vprojection,[4096,  

6.1.2LLMEngineBaselines  

4096]forOprojection,[11008,4096]and[4096,11008]for  

FFN.Foreach[N,K],theinflectionpointsarefoundbasedWeimplementourFlashDecoding++usingthePytorch-  

onthedecisionflowinFigure8(c).Then,alookuptablebasedfront-endwiththeC++andCUDAbackendfor  

isformed,andeachGEMV/GEMMoperationisexecutedNVIDIAGPUswhileROCmforAMDGPUs.Wecom-  

accordingtocorrespondingimplementationsduringruntime.paretheinferenceperformanceinbothprefillphaseand  

Inthisexample,FastGEMVisadoptedfortheK,Q,VdecodephasewiththefollowingLLMenginebaselines:  

projectionwhenbatchsize=1(M=1)forthedecodeHuggingFace(HF)v4.34.1(Wolfetal.,2020),vLLM  

phase,andourflatGEMMoptimizationisappliedwhenv0.1.7(Kwonetal.,2023),DeepSpeedv0.11.1(Aminabadi  

batchsize=1/inputsequencelength=8forFFN(M=8).etal.,2022),TensorRT-LLMv0.5.0(Vaidyaetal.,2023),1  

OpenPPL(Sensetime,2023a),FlashAttentionv2.3.5(Dao,  

2023)andFlashDecoding(Daoetal.,2023).Thesebase-6EVALUATION  

linesareintroducedinSection7.  

6.1ExperimentsSetup  

6.1.3Models  

WeevaluatetheperformanceofFlashDecoding++ondif-  

ferentGPUswithvariousLargeLanguageModels.WeWeevaluatetheperformanceofFlashDecoding++with  

comparetheperformancewithseveralstate-of-the-artLLMotherLLMinferenceenginesonthreetypicalLargeLan-  

inferenceengines.guageModels:Llama2,OPT,andChatGLM2.Table2  


<!-- page 9 -->


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

FlashDecoding++:FasterLargeLanguageModelInferencewit  



HFTensorRT-LLMDeepSpeedPPLFlashAttention2OursOurs-FTL  

batchsize=1248*Ours-FTL:Ours(firsttokenlatency[ms])  

2.02.E+04  

pLu1.E+04d1.0aetee5.E+03npcSy  

0.00.E+00  

1k8k32k1k8k32k1k8k32k1k8k  

**(a)Llama2-7B@A100**  

1.52.E+04  

p1.01.E+04Luadteene0.55.E+03cpyS  

0.00.E+00  

1k8k32k1k8k32k1k8k1k8k  

**(b)Llama2-13B@A100**  

1.56.E+03  

pu1.04.E+03L  

daeteenp0.52.E+03cSy  


| Ours Ours (token/s) |         |
| ------------------- | ------- |
| 28 1k 4k 128 1k 4k  | 128 1k  |
| 1k 2k 128 5         | 12 1    |
| 2-7B (b)Lla         | ma2-13B |
| 28 1k 4k 128 1k 2k  | 128 1k  |
|                     |         |
|                     |         |
|                     |         |

0.00.E+00  

1k8k32k1k8k1k8k1k8k  

**(c)ChatGLM2-6B@A100**  

1.54.E+03  

p3.E+03u1.0Ldae2.E+03teenp0.5c1.E+03yS  

0.00.E+00  

1k8k1k8k1k1k  

**(d)Llama2-7B@3090**  

1.54.E+03  

p3.E+03Lu1.0adte2.E+03ee0.5np1.E+03cSy  

0.00.E+00  

1k8k1k8k1k1k  

**(e)ChatGLM2-6B@3090**  

Figure10.SpeedupoftheprefillphaseonNVIDIAGPUs,normal-  

izedtoFlashAttention.Blankbarsrepresentfailedexecution:(1)  

HuggingFace,DeepSpeedandTensorRT-LLMrunoutofmemory  

withlongsequences.(2)vLLMdoesnotsupportChatGLM2-6B.  

(3)TensorRT-LLMfailstocompileonRTX3090GPUswith24GB  

memory,andfailstocompileforChatGLM2-6Bwithsequence  

length>=8k.(4)pplonlysupportsLlama2models.  

showsthedetailedconfigurationofthesemodels.Notethat  


| ×,1.44×,1.13×,1.24×, |        |
| -------------------- | ------ |
| comparedwithFla      | shDec  |
| llphase,FlashDec     | oding+ |

theremaybeseveralmodelsinoneLLM(e.g.,Llama2-7B,  

e.g.,Llama2-13B)withdifferentconfigurations(numberof  

headsandlayers).  



•Llama2(Touvronetal.,2023)isamainstreamopen-  

sourceLLMsetreleasedbyMetain2023.Itisa  

codingis1.05×,1.06×,1.08×,1.09×,and1.08×,respec-collectionofpretrainedandfine-tunedgenerativetext  

tively.Forprefillphase,FlashDecoding++performsworsemodelsranginginscalefrom7Bto70Bparameters.  

thansomebaselineswithshortsequencesbutalwaysgains  

•OPT(Zhangetal.,2022),isasuiteofdecoder-onlyspeedupwithlongsequences.Thereasonisthat,forprefill  

pre-trainedtransformersrangingfrom125Mto175Bphase,weonlyoptimizetheattentionoperation,andtheat-  

parametersreleasedbyMetaAI.tentionoperationoccupiesmoreofthelatencyassequence  

lengthgrows.  

•ChatGLM2(Duetal.,2022)isanopen-sourceLLM  

WealsoshowthedecoderesultsontwoAMDGPUs.Cur-supportingbilingual(Chinese-English)chat.  

rently,onlyHuggingFaceandvLLMcanbeexecutedon  

AMDGPUsasthebaselines,andvLLMdoesnotsupport  

6.2ComparisonwithState-of-the-art  

RX7900XTXyet.FlashDecoding++achievesupto2.41×  

WecompareFlashDecoding++withstate-of-the-artLLMand4.35×comparedwithHuggingFaceonRX7900XTX  

inferenceenginesinFigure9andFigure10onNVIDIAandMI210,respectively.AndonMI210,theaveragespeed  

GPUs,Figure11andFigure12forAMDGPUs.FortheofFlashDecoding++comparedtovLLMis1.86×.  


<!-- page 10 -->

FlashDecoding++:FasterLargeLanguageModelInferencewithAsynchronization,FlatGEMMOptimization,andHeuristics  



ention2Ours84]  


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



1281k8k32k1281k8k32k  

0.81  

124816124816M  

Figure15.SpeedupovercuBLASwithflatGEMMoptimization.  



beappliedinspecificdomains.Forbothmodels,thein-1281k8k32k1281k8k32k  

putstothesoftmaxoperationareobtainedthroughmultiplebatchsize=4batchsize=8  

datasets.99%ofthesoftmaxinputinCodeLlama-7BrangesFigure13.Benefitsofasynchronizedsoftmax(prefillphase).  

from-0.25to17.6,whilethatofVicuna-7Brangesfrom  

HFFlashDecodingxformersxformers-decoderOursOurs (w/o async)-0.8to9.8.Thus,theasynchronizedsoftmaxmethodisalso  

8applicabletothosefine-tunedmodels.p**@TeslaA100GPU@RTX3090GPU**u6  

de4ep2S6.3.2FlatGEMMOptimization  

0GeoGeocache 6410242048641024204864102420486410242048MeanMeanlengthBenefits.WetestourflatGEMMkernelperformancewithbatchsize=1batchsize=4batchsize=1batchsize=4  

1.5state-of-the-artGEMMlibrary,cuBLASontwoNVIDIApu1deGPUs.TheversionofcuBLASisCUDA11.8.Wevarye0.5pSMfrom1to16todemonstratetheflatGEMMoperation  


|     |
| --- |
|     |

0cache GeoGeo4k32k64k4k32k64k4k32k64k4k32k64kinLLMinference,andeight[K,N]configurationsusedinlengthMeanMean  

batchsize=1batchsize=4batchsize=1batchsize=4threeLLMs(Llama2-7B,OPT-6.7B,andChatGLM2-6B)  

Figure14.Benefitsofasynchronizedsoftmax(decodephase).aredepictedinFigure15.TheflatGEMMoptimizationin  

FlashDecoding++achievesanaverageof7%and17%(up  

6.3AblationStudiesto52%)speeduponTeslaA100andRTX3090,respectively.  

LibrariesincludingcuBLASaredesignedforgeneralpur-  

6.3.1AsynchronizedSoftmaxComputation  

pose,hencenotthebestfortheflatGEMMpractice.The  

Benefits.Theasynchronizedsoftmaxschemecanbeap-speedupis9%and23%forsmallM(i.e.,1and2),show-  

pliedtoboththeprefillphaseandthedecodephase.WetestingthattheproposedflatGEMMoptimizationexploresthe  

theproposedschemeagainststate-of-the-artattentionimple-computationcapabilitywithsmallbatchsizes.  

mentationsinFigure13andFigure14onNVIDIAGPUs.  

Scalability.Theusageofdoublebufferingwithlargesizein  

Fortheprefillphase,FlashDecoding++achieves1.52×and  

N-dimensionislimitedbythesharedmemory(L1cache)  

1.19×averagespeedupcomparedwithxformers(Lefaudeux  

sizeofGPUs.TheresultsinFigure15demonstratethat  

etal.,2022)andFlashAttetion2.Forthedecodephase,  

thestrategyworkswithbothNVIDIATeslaA100GPUs  

FlashDecoding++outperformsthedecoding-tailoredim-  

(192KBL1cacheperSM)andNVIDIARTX3090GPUs  

plementationofxformers(denotedasxformers-decoderin  

(128KBL1cacheperSM)thankstothelargeL1datacache.  

Figure14)withshortKVcachelength,andachievesupto  

ButforAMDGPUs,doublebufferingfailstobenefittheflat  

2.02×speedupoverFlashDecodingwithlongcontext.  

GEMMperformanceduetoalimitedL1datacache(16KB  

Correctness.TheabsolutedifferencebetweentheproposedperCUforAMDMI210).Withoutdoublebuffering,the  

attentionmethodandPyTorchisaverage99.7%<1e-2,flatGEMMoptimizationperformsbadlyinmanycases.In  

andall<1e-1(FlashAttentionleadsto99.8%<1e-2v.s.fact,onAMDGPUs,wesignificantlyrelyonheuristicsto  

PyTorch).AsmentionedinSec.3,weintroducearecom-achieveperformancegains.  

putationmechanismintotheasynchronizedsoftmax,which  

automaticallyselectsFlashAttentionforcomputationwhen6.3.3BenefitsofHeuristicDataflow  

theintermediateresultsoverflow.Thefrequencyofrecom-  

Wetestspeedupofthedecodephasebyadoptingthe  

putationisstatisticallyobtainedtobe0.45%onaverage  

heuristicdataflowinthreeLLMs(Llama2-7B,OPT-6.7B,  

acrossdatasetsincludingARC(Clarketal.,2018),Hel-  

andChatGLM2-6B)onNVIDIAGPUs,andtwoLLMs  

laSwag(Zellersetal.,2019)andWinogrande(Sakaguchi  

(Llama2-7B,OPT-6.7B)onAMDGPUs.Theinputlength  

etal.,2019).  

issetto1024,andtheresultsareshowninFigure16.The  

Scalability.Weextendourapproachtomodelsincludingheuristicdataflowachievesanaverageof10%and20%(up  

CodeLlama-7B(Rozie`reetal.,2023)andVicuna-7B(Chi-to29%)speeduponTeslaA100andRTX3090,respectively.  

angetal.,2023),whicharefine-tunedonLlama2-7BtoOnAMDGPUs,theextensionofFastGEMVimplementa-  


<!-- page 11 -->

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

sionofFlashAttentionandenhancestheparallelismthrough  

splitingKandV,supportingefficientself-attentioncompu-  


| Tesla A10 |     |     |
| --------- | --- | --- |
|           |     |     |

tationforlongsequenceduringthedecodephase.Faster-  

Transformer(NVIDIA,2017b)andOpenPPL(Sensetime,  

2023a)implementlargemodelinferenceenginesusing  

C++toreduceoverheadresultingfromkernelsschedul-  

ing,comparedtoPythonimplementations.Theyalsoem-  

ploymemorymanagementtechniquesandkernelfusionto  

achieveefficientLLMinference.TensorRT-LLM(Vaidya  

etal.,2023)isbuiltupontheTensorRT(NVIDIA)andthe  

FasterTransformer(NVIDIA,2017b)engine(C++)andin-  

corporatescutting-edgeopen-sourcetechnologiessuchas  

FlashAttention(Daoetal.,2022;Dao,2023).Additionally,  

Figu0re16.Benefitsoftheheuristicdataflow(inputlength=102745).itenhancesitseaseofusebyprovidingthePythonAPI.1285121k2k4k  



Hugging FaceDeepSpeedOursOurs (token/s)8CONCLUSION  

490  


|     |     |     |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|     |     |     |     |     |     |     |     |     |     |     |
|     |     |     |     |     |     |     |     |     |     |     |
|     |     |     |     |     |     |     |     |     |     |     |

pu3d85WeproposeFlashDecoding++,afastLargeLanguage2ee80p1Modelinferenceengineinthispaper.FlashDecoding++S075  

1285121k2k4kacceleratesmainstreamLLMswithmultiplehardwareback-  

Figure17.PerformanceonLlama2-13BontwoTeslaA100GPUs.endsupport.FlashDecoding++proposesthreenovelde-  

signs:theasynchronizedsoftmaxwithunifiedmaxvalue,  

tionprovestobehighlyefficient,andleadstosignificanttheflatGEMMoptimizationwithdoublebuffering,and  

performancegainswithsmallbatchsizes.Theaveragetheheuristicdataflowwithhardwareresourceadaption,  

speedupofusingheuristicsis57%and37%onMI210andachievingupto4.86×and4.35×speeduponNVIDIA  

RX7900XTX,respectively.andAMDGPUscomparedwithHuggingFaceimplementa-  

tions.FlashDecoding++alsoachievesanaverageof1.37×  

6.4Multi-GPUsPerformancespeedupcomparedwithstate-of-the-artLLMinferenceen-  

gines,FlashDecoding,onvariousLLMs.FlashDecoding++supportsexecutinglargeLLMsonmul-  

tipleGPUs.WeuseLlama2-13Brunningon2NVIDIA  

TeslaA100GPUstoevaluatetheperformanceofFlashDe-REFERENCES  

coding++.TheresultinFigure17showsthat,FlashDe-  

Textgenerationinference:Fastinference  

coding++achieves2.48×and1.19×higherdecodephase  

optimizeforllms.[Online],2023.  

throughputcomparedwithHuggingFace(Wolfetal.,2020)  

https://github.com/huggingface/  

andDeepSpeed(Aminabadietal.,2022).  

text-generation-inference/.  

7RWELATEDORKSMlcllm:Machinelearningcompilationforlargelanguage  

https://github.com/models.[Online],2023.  

Largelanguagemodelinferenceaccelerationhasgainedmlc-ai/mlc-llm.  

significantattentioninrecentresearch,withseveralnotable  

approachesandtechniquesemerginginthefield.Deep-AMD.Toolstotranslatecudasourcecodeintoportablehip  

Speed(Aminabadietal.,2022)isacomprehensiveenginec++automatically.[Online],2023.https://github.  

thatoptimizesboththetrainingandinferencephasesforcom/ROCm-Developer-Tools/HIPIFY.  

LLMs.Itachievesrobustinferenceperformancethrough  

kernelfusionandefficientGPUmemorymanagement,withAminabadi,R.Y.,Rajbhandari,S.,Awan,A.A.,Li,C.,  

aparticularfocusonoptimizingmemoryusageforKV-Li,D.,Zheng,E.,Ruwase,O.,Smith,S.,Zhang,M.,  

cache.vLLM(Kwonetal.,2023)improvesGPUmemoryRasley,J.,etal.Deepspeed-inference:enablingefficient  

utilizationbyefficientmemorymanagementtechniquesandinferenceoftransformermodelsatunprecedentedscale.  

thePageAttentionmethod,leadingtoincreasedmaximumInSC22:InternationalConferenceforHighPerformance  

batchsizesandelevatingtheupperlimitofinferenceper-Computing,Networking,StorageandAnalysis,pp.1–15.  

formance.FlashAttention(Daoetal.,2022;Dao,2023)IEEE,2022.  

optimizestheself-attentioncomputationprocessduringthe  

prefillphasethroughimprovedparallelismandworkloadAnil,R.,Dai,A.M.,Firat,O.,Johnson,M.,Lepikhin,D.,  

distribution.FlashDecoding(Daoetal.,2023)isanexten-Passos,A.,Shakeri,S.,Taropa,E.,Bailey,P.,Chen,Z.,  


<!-- page 12 -->

FlashDecoding++:FasterLargeLanguageModelInferencewithAsynchronization,FlatGEMMOptimization,andHeuristics  

Chu,E.,Clark,J.H.,Shafey,L.E.,Huang,Y.,Meier-modelsbeyondafixed-lengthcontext.arXivpreprint  

Hellstern,K.,Mishra,G.,Moreira,E.,Omernick,M.,arXiv:1901.02860,2019.  

Robinson,K.,Ruder,S.,Tay,Y.,Xiao,K.,Xu,Y.,Zhang,  

Dao,T.Flashattention-2:Fasterattentionwithbet-Y.,Abrego,G.H.,Ahn,J.,Austin,J.,Barham,P.,Botha,  

terparallelismandworkpartitioning.arXivpreprintJ.,Bradbury,J.,Brahma,S.,Brooks,K.,Catasta,M.,  

arXiv:2307.08691,2023.Cheng,Y.,Cherry,C.,Choquette-Choo,C.A.,Chowd-  

hery,A.,Crepy,C.,Dave,S.,Dehghani,M.,Dev,S.,Dao,T.,Fu,D.,Ermon,S.,Rudra,A.,andRe´,C.Flashat-  

Devlin,J.,D´ıaz,M.,Du,N.,Dyer,E.,Feinberg,V.,Feng,tention:Fastandmemory-efficientexactattentionwith  

F.,Fienber,V.,Freitag,M.,Garcia,X.,Gehrmann,S.,io-awareness.AdvancesinNeuralInformationProcess-  

Gonzalez,L.,Gur-Ari,G.,Hand,S.,Hashemi,H.,Hou,ingSystems,35:16344–16359,2022.  

L.,Howland,J.,Hu,A.,Hui,J.,Hurwitz,J.,Isard,M.,It-  

tycheriah,A.,Jagielski,M.,Jia,W.,Kenealy,K.,Krikun,Dao,T.,Haziza,D.,Massa,F.,andSizov,G.  

M.,Kudugunta,S.,Lan,C.,Lee,K.,Lee,B.,Li,E.,Li,Flash-decodingforlong-contextinference.[Online],  

M.,Li,W.,Li,Y.,Li,J.,Lim,H.,Lin,H.,Liu,Z.,Liu,2023.https://crfm.stanford.edu/2023/  

F.,Maggioni,M.,Mahendru,A.,Maynez,J.,Misra,V.,10/12/flashdecoding.html.  

Moussalem,M.,Nado,Z.,Nham,J.,Ni,E.,Nystrom,A.,  

Dong,Z.,Tang,T.,Li,L.,andZhao,W.AsurveyonParrish,A.,Pellat,M.,Polacek,M.,Polozov,A.,Pope,  

arXivlongtextmodelingwithtransformers.arxiv2023.R.,Qiao,S.,Reif,E.,Richter,B.,Riley,P.,Ros,A.C.,  

preprintarXiv:2302.14502.Roy,A.,Saeta,B.,Samuel,R.,Shelby,R.,Slone,A.,  

Smilkov,D.,So,D.R.,Sohn,D.,Tokumine,S.,Valter,  

Du,Z.,Qian,Y.,Liu,X.,Ding,M.,Qiu,J.,Yang,Z.,and  

D.,Vasudevan,V.,Vodrahalli,K.,Wang,X.,Wang,P.,Tang,J.Glm:Generallanguagemodelpretrainingwith  

Wang,Z.,Wang,T.,Wieting,J.,Wu,Y.,Xu,K.,Xu,Y.,autoregressiveblankinfilling.InProceedingsofthe60th  

Xue,L.,Yin,P.,Yu,J.,Zhang,Q.,Zheng,S.,Zheng,AnnualMeetingoftheAssociationforComputational  

C.,Zhou,W.,Zhou,D.,Petrov,S.,andWu,Y.Palm2Linguistics(Volume1:LongPapers),pp.320–335,2022.  

technicalreport,2023.  

DYLANPATEL,A.A.Theinferencecostofsearch  

Bridle,J.Trainingstochasticmodelrecognitionalgorithmsdisruption-largelanguagemodelcostanalysis.[Online],  

asnetworkscanleadtomaximummutualinformation2023.https://www.semianalysis.com/p/  

Advancesinneuralinformationestimationofparameters.the-inference-cost-of-search-disruption.  

processingsystems,2,1989.  

Kwon,W.,Li,Z.,Zhuang,S.,Sheng,Y.,Zheng,L.,Yu,  

Chiang,W.-L.,Li,Z.,Lin,Z.,Sheng,Y.,Wu,Z.,Zhang,C.H.,Gonzalez,J.,Zhang,H.,andStoica,I.Efficient  

H.,Zheng,L.,Zhuang,S.,Zhuang,Y.,Gonzalez,J.E.,memorymanagementforlargelanguagemodelserving  

Stoica,I.,andXing,E.P.Vicuna:Anopen-sourcewithpagedattention.InProceedingsofthe29thSym-  

chatbotimpressinggpt-4with90%*chatgptquality,posiumonOperatingSystemsPrinciples,pp.611–626,  

March2023.URLhttps://lmsys.org/blog/2023.  

2023-03-30-vicuna/.  

Langley,P.Craftingpapersonmachinelearning.InLangley,  

Clark,P.,Cowhey,I.,Etzioni,O.,Khot,T.,Sabharwal,A.,P.(ed.),Proceedingsofthe17thInternationalConference  

Schoenick,C.,andTafjord,O.ThinkyouhavesolvedonMachineLearning(ICML2000),pp.1207–1216,Stan-  

questionanswering?tryarc,theai2reasoningchallenge.ford,CA,2000.MorganKaufmann.  

ArXiv,abs/1803.05457,2018.  

Lefaudeux,B.,Massa,F.,Liskovich,D.,Xiong,W.,  

Clusmann,J.,Kolbinger,F.R.,Muti,H.S.,Carrero,Caggiano,V.,Naren,S.,Xu,M.,Hu,J.,Tin-  

Z.I.,Eckardt,J.-N.,Laleh,N.G.,Lo¨ffler,C.M.L.,tore,M.,Zhang,S.,Labatut,P.,andHaziza,  

Schwarzkopf,S.-C.,Unger,M.,Veldhuizen,G.P.,D.xformers:Amodularandhackabletrans-  

etal.Thefuturelandscapeoflargelanguagemodelshttps://github.com/formermodellinglibrary.  

inmedicine.CommunicationsMedicine,3(1):141,2023.facebookresearch/xformers,2022.  

Cui,C.,Ma,Y.,Cao,X.,Ye,W.,andWang,Z.Re-Merity,S.,Xiong,C.,Bradbury,J.,andSocher,R.Pointer  

ceive,reason,andreact:Driveasyousaywithlargesentinelmixturemodels,2016.  

languagemodelsinautonomousvehicles.arXivpreprint  

Nair,V.andHinton,G.E.RectifiedlinearunitsimprovearXiv:2310.08034,2023.  

restrictedboltzmannmachines.InProceedingsofthe27th  

Dai,Z.,Yang,Z.,Yang,Y.,Carbonell,J.,Le,Q.V.,andinternationalconferenceonmachinelearning(ICML-10),  

Salakhutdinov,R.Transformer-xl:Attentivelanguagepp.807–814,2010.  


<!-- page 13 -->

FlashDecoding++:FasterLargeLanguageModelInferencewithAsynchronization,FlatGEMMOptimization,andHeuristics  

Nerdynav.Up-to-datechatgptstatisticsandusernumbersThirunavukarasu,A.J.,Ting,D.S.J.,Elangovan,K.,Gutier-  

[oct2023].[Online],2023.https://nerdynav.rez,L.,Tan,T.F.,andTing,D.S.W.Largelanguage  

com/chatgpt-statistics.modelsinmedicine.Naturemedicine,29(8):1930–1940,  

2023.  

NVIDIA.Nvidiatensorrt:Ansdkforhigh-performance  

deeplearninginference.[Online].https://Touvron,H.,Martin,L.,Stone,K.,Albert,P.,Almahairi,  

developer.nvidia.com/tensorrt.A.,Babaei,Y.,Bashlykov,N.,Batra,S.,Bhargava,P.,  

Bhosale,S.,etal.Llama2:Openfoundationandfine-  

NVIDIA.Cutlass:Cudatemplatesforlinearalgebrasub-  

tunedchatmodels.arXivpreprintarXiv:2307.09288,  

routines.[Online],2017a.https://github.com/  

2023.  

NVIDIA/cutlass.  

Vaidya,N.,Oh,F.,andComly,N.Optimizinginference  

NVIDIA.Fastertransformer:Abouttransformerrelatedop-onlargelanguagemodelswithnvidiatensorrt-llm,now  

https:timization,includingbert,gpt.[Online],2017b.publiclyavailable.[Online],2023.https://github.  

//github.com/NVIDIA/FasterTransformer.com/NVIDIA/TensorRT-LLM.  

NVIDIA.cublas:Basiclinearalgebraonnvidiagpus.[On-Vaswani,A.,Shazeer,N.,Parmar,N.,Uszkoreit,J.,Jones,  

line],2017c.https://developer.nvidia.com/L.,Gomez,A.N.,Kaiser,Ł.,andPolosukhin,I.At-  

cublas.tentionisallyouneed.Advancesinneuralinformation  

processingsystems,30,2017.NVIDIA.Nvidiatensorcore.[Online],2023.https:  

//www.nvidia.com/en-us/data-center/Wang,S.Fastgemv:High-speedgemvkernels.[Online],  

tensor-cores/.2023.https://github.com/wangsiping97/  

FastGEMV.OpenAI.Openaipricing.[Online],2023.https://  

openai.com/pricing.Wolf,T.,Debut,L.,Sanh,V.,Chaumond,J.,Delangue,C.,  

Moi,A.,Cistac,P.,Rault,T.,Louf,R.,Funtowicz,M.,Pham,A.,Yang,C.,Sheng,S.,Zhao,S.,Lee,S.,Jiang,  

Davison,J.,Shleifer,S.,vonPlaten,P.,Ma,C.,Jernite,B.,Dong,F.,Guan,X.,andMing,F.OpenLLM:Op-  

Y.,Plu,J.,Xu,C.,LeScao,T.,Gugger,S.,Drame,M.,https:eratingLLMsinproduction,June2023.URL  

Lhoest,Q.,andRush,A.Transformers:State-of-the-art//github.com/bentoml/OpenLLM.  

naturallanguageprocessing.InProceedingsofthe2020  

Ramachandran,P.,Zoph,B.,andLe,Q.V.SearchingforConferenceonEmpiricalMethodsinNaturalLanguage  

activationfunctions.arXivpreprintarXiv:1710.05941,Processing:SystemDemonstrations,pp.38–45,Online,  

2017.October2020.AssociationforComputationalLinguistics.  

doi:10.18653/v1/2020.emnlp-demos.6.URLhttps:  

Rozie`re,B.,Gehring,J.,Gloeckle,F.,Sootla,S.,Gat,I.,Tan,  

//aclanthology.org/2020.emnlp-demos.6.  

X.E.,Adi,Y.,Liu,J.,Sauvestre,R.,Remez,T.,Rapin,J.,  

Kozhevnikov,A.,Evtimov,I.,Bitton,J.,Bhatt,M.,Ferrer,Xiao,G.,Tian,Y.,Chen,B.,Han,S.,andLewis,M.Ef-  

C.C.,Grattafiori,A.,Xiong,W.,De´fossez,A.,Copet,J.,ficientstreaminglanguagemodelswithattentionsinks.  

Azhar,F.,Touvron,H.,Martin,L.,Usunier,N.,Scialom,arXivpreprintarXiv:2309.17453,2023.  

T.,andSynnaeve,G.Codellama:Openfoundation  

Zellers,R.,Holtzman,A.,Bisk,Y.,Farhadi,A.,andChoi,  

modelsforcode.arXivpreprintarXiv:2308.12950,2023.  

Y.Hellaswag:Canamachinereallyfinishyoursen-  

Sakaguchi,K.,Bras,R.L.,Bhagavatula,C.,andChoi,Y.tence?InProceedingsofthe57thAnnualMeetingofthe  

Winogrande:AnadversarialwinogradschemachallengeAssociationforComputationalLinguistics,2019.  

atscale.arXivpreprintarXiv:1907.10641,2019.  

Zhang,S.,Roller,S.,Goyal,N.,Artetxe,M.,Chen,M.,  

Sensetime.Openppl:Ahigh-performancedeeplearn-Chen,S.,Dewan,C.,Diab,M.,Li,X.,Lin,X.V.,Mi-  

inginferenceplatform.[Online],2023a.https:haylov,T.,Ott,M.,Shleifer,S.,Shuster,K.,Simig,D.,  

//openppl.ai/home.Koura,P.S.,Sridhar,A.,Wang,T.,andZettlemoyer,  

L.Opt:Openpre-trainedtransformerlanguagemodels,  

Sensetime.Alightandfastinferenceserviceforllm.[On-2022.  

https://github.com/ModelTC/line],2023b.  

lightllm.  

ADLLMDETAILEDATAFLOWIN  

Sheng,Y.,Zheng,L.,Yuan,B.,Li,Z.,Ryabinin,M.,Chen,FlashDecoding++WITHKERNELFUSION  

B.,Liang,P.,Re,C.,Stoica,I.,andZhang,C.Flexgen:  

High-throughputgenerativeinferenceoflargelanguageFlashDecoding++utilizesopen-sourcekernelsandfuses  

modelswithasinglegpu.2023.kernelsinLLMs.Element-wisekernelslikeactivationand  


<!-- page 14 -->

FlashDecoding++:FasterLargeLanguageModelInferencewithAsynchronization,FlatGEMMOptimization,andHeuristics  



V  

M  

E  

neG  

oEtsenmiaomtPcpsooriginalrcoe ulxulotialrValoeRsapcuouNojhnmmmsedNotMdLlama2cattnjESpraraftaaosiSdsiMctmomreMGeV V strprrRV RQKQOO  

KKVUMLEIGS  



atnVmjco+moEe niMlVfusedorrotalorUaMPPhicuELotuN ocnteNGdELlama2SVaeojdSSIdiQRcsil +sGM+ ttpreMa+e+RKVa RurKOrd+  



Figure18.ExampleofkernelfusionofLlama2dataflow.  

positionencodingarefusedwithlinearkernels.WeshowtwotypesofAMDGPUs.However,theflatGEMM  

anexampleofkernelfusionofLlama2dataflowcomparedoptimizationusestheTensorCore,soweneeddifferent  

totheoriginaldataflowinFigure18.implementationapproachesfortheRX7900XTXand  

MI210.GiventhatMI210hasMatrixCores,ahardware  

structuresimilartoTensorCoresforefficientmatrixBIMPLEMENTATIONONAMD  

computation,wemigrateCUDAcodeandadjustthewarp  

DuetothePyTorch’ssupportforAMDGPUs,wecanper-sizeto64tosuitthisGPU.RX7900XTXdoesnothave  

formlargelanguagemodelinferenceonAMDGPUssimilarMatrixCores,preventingdirectcodemigration.Tothis  

towhatwedoonNVIDIAGPUs.Wehaveimplementedend,weusetheWMMAcompilerintrinsics,suchas  

builtinamdgcnwmmaf1616x16x16f16w32andvalidatedtheeffectivenessofourproposedmethodson,  

20%AMDGPUsusingAMDparallelprogramming.AMDparal-todevelopflatGEMMkernelsresultinginspeedup  

thanthetorch.matmulusedinPyTorchontheRX7900XTX.lelprogrammingsharesmanysimilaritieswithNVIDIApar-  

allelprogramming.Theirprogrammingmodelsaredivided  

intogrid,block,warp,andthread.SimilartotheCUDA  

runtimelibraryofNVIDIA,AMDhastheROCmruntime  

library.WecanuseHIPtodevelopkernelsforAMDGPUs.  

HIPhasAPIsthatcloselyresembleCUDAAPIs.Wecan  

easilyportCUDAcodedevelopedforNVIDIAGPUsto  

HIPcodeforAMDGPUsbymodifyingtheAPInamesor  

usingtheHIPIFYtool(AMD,2023).However,architectural  

differencesbetweenGPUsmeanthatefficientkernelsde-  

velopedforNVIDIAGPUsmaynotnecessarilybeefficient  

onAMDGPUs,andinsomecases,theymaynotevenrun.  

Forexample,consumer-levelGPUliketheRX7900XTX,  

basedontheRDNA3architecture,lacksstructuressimilar  

totheTensorCoreandcannotefficientlyperformmatrix  

operationsusingWMMAinstructionsasCUDA.Incontrast,  

compute-levelGPUliketheMI210,basedontheCDNA2  

architecture,hastheMatrixCorebutwithawarpsizeof  

64,unlikeNVIDIAGPUs.Thisnecessitatesoptimizations  

tailoredforeachoftheseGPUs.  

Weemploydifferentstrategiesforourimplementations  

onthesetwotypesofAMDGPUstoaccommodate  

theirdistinctcharacteristicscomparedtoNVIDIA  

GPUs.Sinceourasynchronizedsoftmaxoptimization  

fordecodephasedoesnotusetheTensorCore,we  

migrateCUDAcodestoHIPandrunthemonthese  
