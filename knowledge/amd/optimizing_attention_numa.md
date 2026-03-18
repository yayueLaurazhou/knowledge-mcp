# Optimizing Attention on GPUs by Exploiting GPU Architectural NUMA Effects

*Author: Mansi Choudhary; Karthik Sangaiah; Sonali Singh; Muhammad Osama; Lisa Wu Wills; Ganesh Dasika*

---


<!-- page 1 -->


#### OAGPUEGPUAPTIMIZINGTTENTIONONSBYXPLOITINGRCHITECTURAL


#### NUMAEFFECTS



MansiChoudhary1*KarthikSangaiah2SonaliSingh2MuhammadOsama2LisaWuWills3GaneshDasika2  

1DepartmentofECE,DukeUniversity,Durham,USA  

2AdvancedMicroDevicesInc.,SantaClara,USA  

3DepartmentofComputerScience,DukeUniversity,Durham,USA  

5  

2ABSTRACT  

0TheriseofdisaggregatedAIGPUshasexposedacriticalbottleneckinlarge-scaleattentionworkloads:non-  

2  

uniformmemoryaccess(NUMA).Asmulti-chipletdesignsbecomethenormforscalingcomputecapabilities,  

vmemorylatencyandbandwidthvarysharplyacrosscomputeregions,underminingtheperformanceoftraditional  

oGPUkernelschedulingstrategiesthatassumeuniformmemoryaccess.WeidentifyhowtheseNUMAeffects  


### N

distortlocalityinmulti-headattention(MHA)andpresentSwizzledHead-firstMapping,aspatially-aware  

schedulingstrategythatalignsattentionheadswithGPUNUMAdomainstoexploitintra-chipletcachereuse.On3  

AMD’sMI300Xarchitecture,ourmethodachievesupto50%higherperformanceoverstate-of-the-artattention  

]algorithmsusingconventionalschedulingtechniquesandsustainsconsistentlyhighL2cachehitratesof80-97%.  


### R

TheseresultsdemonstratethatNUMA-awareschedulingisnowfundamentaltoachievingfullefficiencyon  


### A

next-generationdisaggregatedGPUs,offeringapathforwardforscalableAItrainingandinference.  

.  

s  

c  

1I[NTRODUCTIONwherememoryaccesslatenciesandbandwidthcharacter-  

isticsvarysignificantlybasedonthespatialrelationship  

1ThegrowthofmachinelearningandgenerativeAIhascre-  

betweencomputeunitsandmemoryresources.Chiplet-vatedunprecedenteddemandforhighlyefficientcompute  

baseddesignsrepresentanaturalevolutioninthistrajectory,2  

systems.Asmodelsscaletobillionsofparameters,trainon  

3offeringimprovedmanufacturingyieldsandenhancedscala-  

increasinglylargedatasets,andhandlemassiveinference1bilitywhilemakingNUMAeffectsparticularlypronounced  

workloads,thecomputationalrequirementshavegrownex-2duetointer-diecommunicationoverhead.  

0ponentially,drivingtheneedformorepowerfulandeffi-  

.cientAIaccelerators.ThisrelentlessscalingpressurehasTheTransformerarchitecture(Vaswanietal.,2017)has1  

1fundamentallyreshapedhowMLacceleratorarchitectures,enabledremarkableadvancesinlargelanguagemod-  

5includingGPUs,aredesignedandmanufactured.els(LLMs)andgenerativeAI.TheAttentionmechanism,  

2thefundamentalcomponentofTransformers,hasbecome  

:Tomeetthesescalingdemandswhilemanagingmanufactur-  

vaprimarybottleneckingenerativeAIworkloadsduetoits  

ingcostsandyields,AIacceleratorarchitectures,includingiquadratictimeandmemorycomplexityassequencelengths  

XAIGPUs,haveevolvedtowardincreasinglydisaggregatedgrow.Thisbottleneckhasfueledextensiveoptimization  

rdesigns.ModernAIGPUsfeaturedistributedmemoryhier-  

aeffortsoverrecentyears(Daoetal.,2022;Dao,2024;Shah  

archieswithmultiplememorycontrollersandcachestruc-etal.,2025;Sanovaretal.,2025),withapproachesrang-  

turesspreadacrosslargedieareasormultiplechiplets.Thisingfromalgorithmicimprovements(Daoetal.,2022;Dao,  

disaggregation,whetherthroughlargermonolithicdieswith2024;Sanovaretal.,2025)tohardware-specificoptimiza-  

distributedresourcesorchiplet-basedmulti-dieintegration,tions(Shahetal.,2025).  

fundamentallyalterstheorganizationofthememorysub-  

system.UnlikeearlierGPUgenerationswithrelativelyHowever,existingattentionoptimizationslargelyassume  

uniformmemoryaccesspatterns,thesescaledarchitecturesuniformmemoryaccesspatternsanddonotaccountforthe  

introducenon-uniformmemoryaccess(NUMA)effects,NUMAeffectspresentinmodernscaledGPUarchitectures.  

RatherthanviewingNUMAcharacteristicsasobstaclesto  

*WorkcompletedduringaninternshipatAd-overcome,werecognizethemascreatingnewopportuni-  

vancedMicroDevicesInc.Correspondenceto:Mansi  

tiesforperformanceoptimizationthroughspatially-aware  

Choudhary<mc846@duke.edu>,KarthikSangaiah  

kerneldesign.Todemonstratethisapproach,wefocusour<karthik.sangaiah@amd.com>.  

studyonAMD’sMI300Xarchitecture,whichrepresents  


<!-- page 2 -->

Die 0Die 0Die 2  


| CU CU CU CU
CU CU CU CU
CU CU CU CU
Shared L2 |
| --------------------------------------------- |
| Memory Controller
HBMs                        |


| CU  | CU  | CU  | CU  |
| --- | --- | --- | --- |
| CU  | CU  | CU  | CU  |
| CU  | CU  | CU  | CU  |


| Die 0
CU CU CU
2L CU CU CU
CU CU CU
Interconnect | rellortnoC
meM
sMBH
rellortnoC
meM |
| ------------------------------------------------ | ---------------------------------- |
| CU CU CU
2L CU CU CU
CU CU CU
Die 1              |                                    |


| rellortnoC
meM
sMBH
rellortnoC
meM | Die 0
CU CU
2L CU CU
CU CU
Interconnect | Die 2
CU CU tcennocretnI
2L CU CU
CU CU
Interconnect | rellortnoC
meM
sMBH
rellortnoC
meM |
| ---------------------------------- | --------------------------------------- | ---------------------------------------------------- | ---------------------------------- |
|                                    | CU CU
2L CU CU
CU CU
Die 1              | CU CU tcennocretnI
2L CU CU
CU CU
Die 3              |                                    |


| Die 0
CU |
| -------- |
| CU       |


| Die 2
CU |
| -------- |
| CU       |

Multi-Die Chiplet  


| CU  | CU  | CU  |
| --- | --- | --- |

Architecture  



Memory  


| CU  | CU  | CU  |
| --- | --- | --- |
| CU  | CU  | CU  |


| CU  |
| --- |
| CU  |


| CU  |
| --- |
| CU  |

Disaggregation  



(a) Single-Die GPUs(b) Dual-Die GPUs(c) Quad-Die GPUs  



Figure1.EvolutionofGPUarchitecturestowarddisaggregatedmemoryhierarchies.(a)Traditionalsingle-dieGPU(e.g.,NVIDIA  

A100(NVIDIA,2020),H100(NVIDIA,2022);AMDMI200series(AMD,2021))withunifiedL2cachesharedacrossallcomputeunits  

(CUs),providinguniformmemoryaccess.(b)Dual-diechipletarchitecture(e.g.,NVIDIABlackwell(NVIDIA,2024a),Rubin(NVIDIA,  

2024b))withinterconnectsbetweendies.(c)Quad-diechipletarchitecture(e.g.,NVIDIARubinUltra(NVIDIA,2024c),AMD  

MI300(AMD,2023)series)withfurtherdisaggregation.Eachdiehasdedicatedcomputeunits,L2cache,andmemorycontrollers  

connectedtoHBMstacks.WhileAMDandNVIDIAbothemploymulti-diedesigns,thedegreetowhichNUMAeffectsareexposedto  

softwarevariesbyimplementation.NVIDIA’sBlackwellmaintainsfullcachecoherencybetweenthedies,abstractingtheNUMAeffects  

atthehardwarelevel,whereasAMD’sMI300XexplicitlyexposesNUMAcharacteristics,enablingarchitecture-awareoptimizations.  



oneofthemostadvancedimplementationsofchiplet-basedInthiswork,wepresentthefollowingcontributions:  

designincurrent-generationAIaccelerators.TheMI300X  

featureseightXCDs(AcceleratorComplexDies),eachwith•AnalysisofNUMAeffectsinchiplet-basedmulti-die  

dedicatedcomputeunits,L2cache,andmemorycontrollersGPUarchitecturesandtheirimplicationsforkernel  

connectedtoindependentHBMstacks.Thismulti-chipletperformance  

configurationwithdistributedmemorysubsystemsmakes  

NUMAeffectsparticularlypronounced,providinganideal•Characterizationofspatiallocalitypatternsinherent  

testbedfordevelopingandvalidatingspatially-awareopti-inattentionmechanisms(forwardandbackwardvari-  

mizationtechniques.Asdisaggregatedandchiplet-basedants)andidentificationofopportunitiesformemory  

architecturesbecomeincreasinglyprevalentacrossthein-hierarchyoptimization  

dustryforscalingAIworkloads,insightsgainedfromop-  

•DesignandimplementationofSwizzledHead-firsttimizingforsuchdesignswillhavebroadapplicabilityto  

Mapping,anovelspatially-awareattentionoptimiza-futuregenerationsofGPUsandotherAIaccelerators.  

tiontechniquethatalignscomputationalworkwith  

ThisfocusonNUMA-awareoptimizationbuildsuponpriorNUMAboundariestomaximizecacheefficiency  

workdemonstratingsignificantbenefitsfromarchitecture-  

awarekerneldesign.Forinstance,spatially-awaremapping•Comprehensiveperformanceevaluationcomparing  

strategiesforGEMMoperationsontheMI300Xhaveshownspatially-unawareapproachestoourproposedmap-  

dramaticimprovementsincacheutilization,withL2cacheping,demonstratingupto50%higherperformance  

hitratesincreasingfrom43%to92%whenaccountingforoverstate-of-the-artattentionalgorithmsusingconven-  

NUMAcharacteristics(AMD,2025c).Theseresultsunder-tionalschedulingtechniquesandconsistentlysustain-  

scorethecriticalimportanceofarchitecture-awareoptimiza-inghighL2cachehitratesof80-97%.  

tionandmotivateourinvestigationofsimilartechniquesfor  

attentionkernels,whichrepresentanothercriticalbottleneck2BACKGROUND  

inmodernAIworkloads.  

2.1NUMAEffectsinModernGPUArchitectures  

Inthispaper,weintroduceSwizzledHead-firstMapping,  

anovel,spatially-awaretechniquethatexploitstheinher-ModernGPUarchitecturesdesignedforAIworkloadshave  

entspatiallocalitypatternsinAttentioncomputationstoincreasinglyadopteddisaggregatedmemoryhierarchiesto  

mitigateNUMAeffectsandmaximizecacheutilizationinachievethenecessaryscaleincomputecapacityandmem-  

chiplet-basedGPUarchitectures.Ourapproachbuildsuponorybandwidth.Whetherthroughlargemonolithicdesigns  

thesuccessofspatially-awareGEMMoptimizationsbyiden-withdistributedresourcesorchiplet-basedmulti-dieinte-  

tifyingandexploitinganalogouslocalitypatternswithinthegration,thesearchitecturesdepartfromtraditionalunified  

Attentionmechanism.cachedesignsinwhichallcomputeunitssharecommon  

cachestructures.ThisdisaggregationintroducesNUMA  

2  


<!-- page 3 -->


| HBMs + Last Level Cache (LLC)                                 |
| ------------------------------------------------------------- |
|                                                               |
| L2
CU0 warms up
Tile the cache with
first-touch
CU0 CU1
Die 0 |

effects,wherememoryaccesslatenciesandbandwidthchar-  

acteristicsvarybasedonthespatialrelationshipbetween  


| L2
CUs on Die 1 get
no reuse out of
Tile Die 0’s L2 cache
and need to load
again from HBM
CU2 CU3
Die 1 |          |     |                                                                                   |     |
| ------------------------------------------------------------------------------------------------------- | -------- | --- | --------------------------------------------------------------------------------- | --- |
|                                                                                                         |          |     | CUs on Die 1 get
no reuse out of
Die 0’s L2 cache
and need to load
again from HBM |     |
|                                                                                                         | Tile
CU2 |     |                                                                                   |     |
|                                                                                                         |          |     |                                                                                   |     |

computeunitsandthememoryresourcestheyaccess.  

NUMAeffectsmanifestasahierarchyofmemoryaccess  


| Tile
CU0 |     | C
t | U0 warms up
he cache with
first-touch |     |
| -------- | --- | --- | ------------------------------------- | --- |
|          |     |     | CU1                                   |     |


| o reuse out o
ie 0’s L2 cach
nd need to loa
gain from HB |
| -------------------------------------------------------- |
| CU3                                                      |

costswithintheGPU.Localaccesses,wherecomputeunits  

accessmemoryresourcesphysicallyproximatetothem,ben-  

efitfromhigh-bandwidth,low-latencypathsthroughnearby  

privatecaches.Incontrast,remoteaccessesthatmusttra-  

verselongerdistancesorcrossinterconnectboundariesex-  

periencesubstantiallyhigherlatenciesandreducedeffective  

bandwidth.Furthermore,distributedcachehierarchiesmean  

thatdatacachedinoneregionprovidesnobenefittocom-Figure2.Impactofworkgroupschedulingoncachereuseinthe  

puteunitsinotherregions,effectivelyfragmentingthetotalTMmulti-diechipletarchitectureofAMDMI300X.Whenwork-  

cachecapacity.Workloadsthatfailtoaccountforthesegroupsprocessingrelatedtilesthatshareinputdataaresched-  

spatialcharacteristicssufferfromreducedcachehitratesuledtodifferentdies(left:CU0onDie0,right:CU2onDie1),  

andpoormemorybandwidthutilization.theycannotbenefitfromeachother’scacheddata.Thiscross-die  

schedulingforcesredundantmemoryfetchesfromHBMthrough  

Thistransitionfromunifiedtodistributedmemoryhier-thesharedlast-levelcache(LLC),asL2cachesareprivatetoeach  

archiesfundamentallychangesoptimizationprioritiesforAcceleratorComplexDieorXCD.TheNUMAeffectsinherentto  

GPUkernels.Intraditionalarchitectureswithuniformac-thisdisaggregatedarchitecturemakespatiallyawareworkgroup  

cesspatterns,temporallocalitywastheprimaryoptimiza-placementcriticalformaximizingcacheefficiencyandminimizing  

memorybandwidthconsumption.tiontarget,withthefocusonwhenworkisscheduledto  

maximizecachereuseovertime.ArchitectureswithNUMA  

effectsintroducespatialdimensionstothisoptimization  

improvinglocality.space,wherebothwhenandwhereworkisscheduledbe-  

comeequallycritical.Performancebecomeshighlysensi-Thisremappingapproachgeneralizesthecooperativethread  

tivetothespatialdistributionofcomputationalworkandarray(CTA)orworkgroupswizzlingtechniqueswidely  

dataplacement,requiringkerneldesignerstoexplicitlycon-usedinhigh-performanceGEMMlibrariessuchasTen-  

siderthephysicaltopologyoftheunderlyinghardwaretosile(AMD,2025c),hipBLASLt(AMD,2025b),andTri-  

achieveoptimalperformance.ton(TritonDevelopers,2025).Intheselibraries,swizzling  

transformsthedefaultlinearworkgroupassignmentinto  

2.2CurrentMitigationsforSpatialNUMAEffectspatternsthatbetteralignwithmemorylayoutsorhardware  

topology.Forinstance,matricesmaybetiledinto2Dblocks  

Modernmulti-diechipletGPUsscheduleworkgroups  

tomaximizedatareusewithinsharedcaches.Whenap-  

(WGs)acrosscomputediesusingachunkedround-robin  

pliedtoworkloadsonmulti-diearchitectures,suchremap-  

policy,whereeachdie(orNUMAdomain)receivesasmall,  

pingalignsthelogicalcomputationorderwiththephysical  

contiguousbatchofWGsbeforethescheduleradvancesto  

NUMAtopology,mitigatingscheduler-inducedinefficien-  

thenext.Oncurrenthardware,thischunksizeissettoone  

cieswhilemaintainingportabilityacrossfuturehardware  

(seeFigure2),ensuringloadbalanceandfullutilizationof  

schedulers.Inthiswork,weextendthesetechniquesto  

aggregateHBMbandwidth.However,thisdefaultpolicy  

implementspatially-awaremappingpatternsforAttention  

caninadvertentlyfragmentspatiallocalitywhenadjacent  

onInstinctGPUs.  

WGsoperateonneighboringregionsofthesametensorbut  

aredispatchedtodifferentdies.  

2.3TheAttentionMechanism  

Becausethismappingstrategyisimplementedinthedriver  

Attention(Vaswanietal.,2017)isthecentralcomponentinandsubjecttochangeacrossGPUgenerations,programmers  

theTransformerarchitecturethathasenabledtheremarkablecannotrelyonfixedschedulingbehavior.Instead,kernels  

capabilitiesofmoderngenerativeAImodels.Theattentionmustincorporatemutable,algorithmicmappinglogicthat  

mechanismallowsmodelstoselectivelyfocusonthemostadaptsatruntime.Onewidelyadoptedtechniquetoaddress  

relevantpartsofaninputsequencewhengeneratingeachthischallengeisswizzling,whichreferstothedeliberate  

outputtoken.Thiscapabilityenableslanguagemodelstoreorderingofhowcomputationalworkismappedtohard-  

maintaincoherentcontextacrosslongsequencesandpro-wareexecutionunitstoimprovememorylocalityandcache  

ducehigh-quality,contextuallyappropriateoutputs.utilization.AsillustratedinFigure3,swizzlingremaps  

workgroupIDssothatspatiallyadjacenttilesareassignedThecoreattentioncomputationinvolvesthreekeymatrices  

tothesamedie,therebyexploitingper-dieL2cachesandderivedfromtheinput:queries(Q),keys(K),andvalues  

3  


<!-- page 4 -->

duringinference.BothMHAandGQAcreatecomputa-  


| @triton.jit()                            |
| ---------------------------------------- |
| def swizzle_chiplet(wgid, grid, NUM_XCD: |
| t1_constexpr):                           |
| # Number of wgids/XCD in the new config  |
| wgids_per_xcd = grid // NUM_XCD          |
|                                          |
| # Compute local wgid within the XCD      |
| xcd = wgid % NUM_XCD                     |
| local_wgid = wgid // NUM_XCD             |
|                                          |
| # New wgid based on the new grouping     |
| new _wgid = xcd * wgids_per_xcd +        |
| local_wgid                               |
| return new_wgid                          |

1  

tionalgridsspanningmultipleattentionheadsandbatch2  

items.Eachgridcelloperatesindependentlywithnodata  

3reusebetweenheadsorbatchitems.Figure5showsthis  

4grid,whereeachheadspansaquerylength(NCTX)and  

5headdimension(HEADDIM).  

6  

7Akeylimitationoftraditionalattentionisitsquadratic  

8timeandmemorycomplexity,especiallyassequencelength  

9  

grows.Thisbecomesabottleneckduetotheneedtostore10  

intermediatetensors(S,P)inglobalmemory.11  

122.3.1FlashAttention  

FlashAttention(FA)(Daoetal.,2022)addressesthislimi-Figure3.Chiplet-awareworkgroupIDremappinginTriton.The  

tationbyusinganonlinesoftmaxapproach,enablingtiledfunctiontransformstheoriginalworkgroupIDbydetermining  

computationwhereintermediateblocksofSandPresideinthetargetXCDandcalculatingthenewglobalworkgroupIDto  

achievespatiallocalitywithinchipletmemorydomains.localmemoryratherthanglobalmemory.Across-tile“fix-  

up”stepensuresnumericalcorrectnesswhilesignificantly  

reducingmemoryoverhead.Thistilingstrategyappliesto  

(V).Giventheseinputsequencesandheaddimensiond,theboththeforwardpass,whichcomputesattentionoutputs,  

attentionoutputOiscomputedthroughathree-stepprocess:andthebackwardpass,whichcomputesgradientswithre-  

specttoQ,K,andVduringtraining.  

S  

TS=QK,P=softmax(√),O=PV.(1)  

FlashAttention2(FA2)(Dao,2024)furtherimprovesper-d  

formancebyparallelizingWGsacrossthecontextlength  

forbothforwardandbackwardcomputations.AsshownInEquation1,thefirststepcomputesattentionscoresSby  

inFigure4,Qispartitionedintorowblocksprocessedintakingthedotproductbetweenqueriesandkeys,measuring  

parallel,onerowblockperWG,witheachaccessingthetherelevanceofeachkeytoeachquery.Thesescoresare  

fullKandVtensors.Duringthebackwardpass,thegradi-  

thennormalizedusingthesoftmaxfunctiontoproduceat-  

entcomputationfollowsasimilartilingpattern,witheach  

tentionweightsP,whichrepresenttherelativeimportance  

WGcomputinggradientsforitsassignedQblockwhileofeachinputposition.Finally,theseweightsareusedto  

accessingthecompleteKandVtensors.Thisdatareusecomputeaweightedaverageofthevaluevectors,producing  

patternacrossbothforwardandbackwardpassesiscentralthefinaloutputO.  

toouroptimizationforchiplet-basedGPUs,asdetailedin  

Duringtraining,gradientsmustbecomputedwithrespecttoSection3.  

allinputs.Giventhegradientofthelosswithrespecttothe  

outputdO,thebackwardpasscomputesgradientsthrough  

3FA2SPATIALLOCALITYANDMAPPING  

theattentioncomputationinreverseorder:  

PATTERNS  

TRN×dTRN×NdV=PdO∈,dP=dOV∈,  

Thissectionexploresthespatiallocalityinherentinthe  

RN×NdS=dsoftmax(dP)∈,(2)FlashAttention2algorithmanddemonstrateshowitcanbe  

leveragedtooptimizeperformanceonchiplet-basedGPURN×dTRN×ddQ=dSK∈,dK=dSQ∈.  

architecture,specificallyAMD’schiplet-basedInstintGPU  

-MI300X.WefirstanalyzeFA2’scomputationalgridstruc-  

Inpractice,Transformermodelsusemulti-headattentiontureandexaminecurrentmethodsformappingthisgrid  

(MHA),whichsplitsthemodeldimension(d)acrossmul-todistributedcomputediesorXCDs.Wethenintroduce  

tipleattentionheads.Eachheadoperatesonaportionofanoveloptimization,SwizzledHead-firstMapping,that  

thefulldimensionalitywithseparatequery,key,andvalueimprovesdatareusebyaligningcomputationalworkwith  

projections,allowingdifferentheadstocapturedistinctrela-theNUMAcharacteristicsofchipletarchitectures,thereby  

tionshipsandattendtovariousaspectsoftheinputsequence.  

enhancingper-XCDcacheutilizationandoverallperfor-  

Morerecentmodelshaveadoptedgroupedqueryattention  

mance.  

(GQA)(Ainslieetal.,2023),whichreducesmemoryre-  

quirementsbysharingkeyandvalueprojectionsacross  

multiplequeryheadswhilemaintainingseparatequerypro-  

jections.Thissignificantlyreducesthememoryfootprint  

4  


<!-- page 5 -->

**N_CTX**  

**B**  


|     |     |     |     | MID_DAEH |
| --- | --- | --- | --- | -------- |
|     |     |     |     |          |
|     |     |     |     |          |
|     |     |     |     |          |
|     |     |     |     |          |


|     |     |     |     |
| --- | --- | --- | --- |

**TK**  

**xHEAD_DIM**  

**QV**  

**MM**  


|     |
| --- |
|     |
|     |
|     |


|     |     |     |     |
| --- | --- | --- | --- |
|     |     |     |     |
|     |     |     |     |
|     |     |     |     |


|     |
| --- |
|     |
|     |
|     |

**__**  

**pid 0KK**  

**CC**  

**OO**  

**LLpid 1BlobalB**  


|     | XTC_N |
| --- | ----- |


|     | XTC_N |
| --- | ----- |

**x=**  

**oftmaxpid 2**  

**pid 3**  

**HEAD_DIMHEAD_DIMS = QKTP = softmax(S)O = P.V**  

**(N_CTX, N_CTX)(N_CTX, N_CTX)(N_CTX, HEAD_DIM)**  



Figure4.FlashAttention2TiledComputePartitioningAcrossWorkgroupsforaSingleAttentionHead.QuerymatrixQispartitioned  

intorowblocks(BLOCKM),witheachworkgroup(pid0-pid3)processingonerowblock.Eachworkgroupaccessesthecompletekey  

matrixKTandvaluematrixVtocomputeattentionscoresS=QKT,applysoftmaxtoobtainattentionweightsP,andproduceoutput  

O=PV.AllworkgroupswithinthesameattentionheadshareaccesstothesameKandVtensors,creatingnaturalspatiallocality  

patterns.Matrixdimensionsareshowninparentheses,whereNCTXisthecontextlengthandHEADDIMistheheaddimension.  


| HQ 0 | HQ 1
K, V | HQ 2
K, V | HQ 3 |
| ---- | --------- | --------- | ---- |
| K, V |           |           | K, V |


| HQ 0     | HQ 1 | HQ 2 | HQ 3 |
| -------- | ---- | ---- | ---- |
| K, V
0 0 |      |      |      |



Batch 1Batch Z  

**X**  

..........**T12H12HC**  

**_N**  

K, VK, VK, VK, V0000112233**HEAD_DIM**  

ACC0ACC1ACC2ACC3ACC0  

Figure5.AttentionAlgorithmGridforQ,K,V,Otensors.(a) Multi Head Attention(b) Grouped Query Attention  

Z=batchsize,H=#ofattentionheads,sizedbyquerycontext  

length(NCTX)andheaddimension(HEADDIM).  

Figure6.Attentioncomputecluster(ACC)organizationintrans-  

formerattentionmechanisms.OperationsinvolvingthesameK  

andVtensorsareco-locatedwithinasingleACCforoptimal  

3.1SpatialLocalityinFlashAttention2  

cacheutilization.(a)Inmulti-headattention,eachheadmaintains  

distinctK,Vtensors(e.g.,DeepSeek-V3),requiringoneACCperFlashAttention2exhibitsinherentspatiallocalitypatternsin  

head.(b)Inagroupedqueryattention,multipleheads(fourinthisbothforwardandbackwardpasses,creatingopportunities  

figure)shareK,Vtensorswithinagroup(e.g.,Llama,Mistral),forcacheoptimizationinchiplet-basedarchitectures.As  

requiringoneACCpergroup.  

illustratedinFigure4,duringtheforwardpass,withina  

singleattentionhead,eachrowblockofthequerymatrix  

Qmustaccessthecompletekey(K)andvalue(V)tensors  

singleheadsharedataduringbothforwardandbackward  

tocomputeattentionoutputs.Thisaccesspatterncreates  

computation,butworkgroupsacrossdifferentheadsoper-  

naturaldata-sharingopportunities,asmultipleworkgroups  

ateonentirelyseparatetensors.Incontrast,GQAextends  

processingdifferentrowblocksofthesameattentionhead  

thissharingpatternbyhavingmultiplequeryheadsshare  

(e.g.,pid0topid3inFigure4)requirethesameKandV  

thesamekeyandvaluetensors,effectivelycreatinglarger  

data.  

groupsofworkgroupsthatallaccessthesameKandVdata  

Thebackwardpassfollowsasimilarspatiallocalitypattern.duringbothforwardandbackwardpasses.  

Duringgradientcomputation,eachworkgroupcomputing  

Fromaspatiallocalityperspective,workgroupsthatshare  

differentrowblocksofthegradientsdQ,dK,anddVmay  

thesameinputtensorsduringeitherforwardorbackward  

sharethesameQ,K,V,anddOtensorswithinthesame  

computationnaturallyformwhatwetermanAttention  

attentionhead.Thisconsistentdatasharingpatternacross  

ComputeCluster(ACC).InMHA,eachACCcorresponds  

bothforwardandbackwardpassesamplifiesthebenefitsof  

toallworkgroupswithinasingleattentionheadforboth  

cache-awaremappingstrategies.  

passes.InGQA,eachACCencompassesallworkgroups  

Theextentofthisdatasharinginboththeforwardandtheacrossthegroupedheadsthatsharethesamekey-valueten-  

backwardpassdependsonthespecificattentionmechanismsorsduringtraining.Thekeyoptimizationinsightisthat  

employed.InMHA,eachattentionheadmaintainsitsownco-locatingallworkgroupswithinanACConthesame  

distinctKandVtensors,meaningworkgroupswithinaXCDmaximizesL2cacheutilizationthroughtwomecha-  

5  


<!-- page 6 -->


| BBlloocckk 00 BBlloocckk 10 0DCX BBlloocckk 00 BBlloocckk 10 0DCX
BBlloocckk 21 BBlloocckk 41 BBlloocckk 21 BBlloocckk 41
HQ 0 HQ 1 2L HQ 2 HQ 3 2L
BBlloocckk 00 BBlloocckk 10 0DCX BBlloocckk 00 BBlloocckk 10 0DCX
BBlloocckk 21 BBlloocckk 41 BBlloocckk 21 BBlloocckk 41
HQ 4 HQ 5 2L HQ 6 HQ 7 2L |              |              |              |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------ | ------------ | ------------ |
| XCD0: HQ 0,1                                                                                                                                                                                                                                                                                            | XCD1: HQ 2,3 | XCD2: HQ 4,5 | XCD3: HQ 6,7 |


| BBlloocckk 00 BBlloocckk 10 0DCX
BBlloocckk 21 BBlloocckk 41 |      |      |     |
| ------------------------------------------------------------ | ---- | ---- | --- |
| 2L                                                           | HQ 0 | HQ 1 |     |

nisms:first,sharedtensordataofanACCisloadedintoa  


| BBlloocckk 00 | BBlloocckk 10 |
| ------------- | ------------- |
| BBlloocckk 21 | BBlloocckk 41 |


| BBlloocckk 00 | BBlloocckk 10 |
| ------------- | ------------- |
| BBlloocckk 21 | BBlloocckk 41 |

singleXCD’sL2cacheandreusedacrossmultiplework-  

groups,preventingcachefragmentationwhileincreasing  

datareuse;second,thisreducesthetotalnumberofmemory  


| HQ 2 | HQ 3 |     |
| ---- | ---- | --- |

fetchesfromHBM,aseachsharedtensorisloadedonlyonce  


| BBlloocckk 00 BBlloocckk 10 0DCX
BBlloocckk 21 BBlloocckk 41 |      |      |     |
| ------------------------------------------------------------ | ---- | ---- | --- |
| 2L                                                           | HQ 4 | HQ 5 |     |


| BBlloocckk 00 | BBlloocckk 10 |
| ------------- | ------------- |
| BBlloocckk 21 | BBlloocckk 41 |


| BBlloocckk 00 | BBlloocckk 10 |
| ------------- | ------------- |
| BBlloocckk 21 | BBlloocckk 41 |

perdeviceratherthanredundantlyacrossmultipleXCDs.  

Thisapproachmirrorssuccessfultilingstrategiesusedin  

GEMMoptimizations,whereco-locatingworkgroupsthat  


| HQ 6 | HQ 7 |     |
| ---- | ---- | --- |

shareinputdatasignificantlyimprovesmemorybandwidth  

utilizationandcomputethroughput.  



3.2CurrentFA2Grid-to-XCDMappingPatterns  

Figure8.SwizzledBlock-firstmapping(eightqheads,128row  

FA2supportsmultiplestrategiesformappingWGstoXCDs.blocks,fourXCDs):Block-firstiterationwithGQA-awareswiz-  

Wecategorizeexistingapproachesbasedontheiriterationzlingtoco-locategroupedheadswithinXCDs.Thisschemeworks  

orderandXCDassignmentstrategy.bestwhenGQAgroupsmatchXCDcount.Eachuniquecolor  

correspondstoauniqueattentionhead.  

3.2.1NaiveBlock-first  

TheNaiveBlock-firststrategyiteratesthroughallattention  

L2cacheefficiency.ThisschemeisdeployedbyAMD’s  

headsforeachrowblockbeforemovingtothenextblock  

AITER(AMD,2025a)repositoryofhigh-performanceAI  

(i.e.,completesblock0acrossallheads,thenblock1across  

operators.  

allheads,etc.).WGsareassignedtoXCDsinround-robin  

fashion(e.g.,XCD0getsblock0ofHQ0(QueryHead0),3.2.3NaiveHead-first  

XCD1getsBlock0ofHQ1,etc).ThissplitsACCsacross  

TheNaiveHead-firststrategyiteratesthroughallrowblocksXCDs,reducingcacheefficiency.Thisschemeisabaseline,  

ofasingleattentionheadbeforemovingtothenextheadun-swizzledversionofthemappinginsection3.2.2.  

(i.e.,completesallblocksofHQ0,thenallblocksofHQ  

1,etc.).Withround-robinXCDassignment,eachhead’s  


| BBlloocckk 00 BBlloocckk 10 0DCX BBlloocckk 00 BBlloocckk 10 0DCX
BBlloocckk 21 BBlloocckk 41 BBlloocckk 21 BBlloocckk 41
HQ 0 HQ 4 2L HQ 1 HQ 5 2L
BBlloocckk 00 BBlloocckk 10 0DCX BBlloocckk 00 BBlloocckk 10 0DCX
BBlloocckk 21 BBlloocckk 41 BBlloocckk 21 BBlloocckk 41
HQ 2 HQ 6 2L HQ 3 HQ 7 2L |              |              |              |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------ | ------------ | ------------ |
| XCD0: HQ 0,4                                                                                                                                                                                                                                                                                            | XCD1: HQ 1,5 | XCD2: HQ 2,6 | XCD3: HQ 3,7 |


| BBlloocckk 00 BBlloocckk 10 0DCX
BBlloocckk 21 BBlloocckk 41 |      |      |     | BBlloocckk 00 BBlloocckk 10 0DCX
BBlloocckk 21 BBlloocckk 41 |      |      |     |
| ------------------------------------------------------------ | ---- | ---- | --- | ------------------------------------------------------------ | ---- | ---- | --- |
| 2L                                                           | HQ 0 | HQ 4 |     | 2L                                                           | HQ 1 | HQ 5 |     |


| BBlloocckk 00 | BBlloocckk 10 |
| ------------- | ------------- |
| BBlloocckk 21 | BBlloocckk 41 |


| BBlloocckk 00 | BBlloocckk 10 |
| ------------- | ------------- |
| BBlloocckk 21 | BBlloocckk 41 |

blocksarestripedacrossallXCDs(Figure9).Whilethis  

maintainshead-levelcoherenceduringiteration,itstillsplits  

individualACCsacrossXCDs.Thisschemeisimplemented  

inTriton’sdefaultFlashAttentionkernel(Tilletetal.,2019).  


| BBlloocckk 00 BBlloocckk 10 0DCX
BBlloocckk 21 BBlloocckk 41 | BBlloocckk 00 BBlloocckk 10 0DCX
BBlloocckk 21 BBlloocckk 41 |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| HQ 2 HQ 6 2L                                                 | HQ 3 HQ 7 2L                                                 |


| BBlloocckk 00 | BBlloocckk 10 |
| ------------- | ------------- |
| BBlloocckk 21 | BBlloocckk 41 |


| BBlloocckk 00 | BBlloocckk 10 |
| ------------- | ------------- |
| BBlloocckk 21 | BBlloocckk 41 |


| BBlloocckk 00 BBlloocckk 14 0DCX BBlloocckk 01 BBlloocckk 15 0DCX
BBlloocckk 28 BBlolocckk 142 BBlloocckk 29 BBlolocckk 143
HQ 0 2L HQ 0 2L
BBlloocckk 02 BBlloocckk 16 0DCX BBlloocckk 03 BBlloocckk 17 0DCX
BBlolocckk 120 BBlolocckk 144 BBlolocckk 121 BBlolocckk 145
HQ 0 2L HQ 0 2L |             |             |             |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------- | ----------- | ----------- |
| XCD0: HQ0-7                                                                                                                                                                                                                                                                               | XCD1: HQ0-7 | XCD1: HQ0-7 | XCD2: HQ0-7 |


| 0DCX    | BBlloocckk 00 | BBlloocckk 14
BBlolocckk 142 | 0DCX    | BBlloocckk 01 | BBlloocckk 15  |
| ------- | ------------- | ---------------------------- | ------- | ------------- | -------------- |
|         | BBlloocckk 28 |                              |         | BBlloocckk 29 | BBlolocckk 143 |
| HQ 0 2L |               |                              | HQ 0 2L |               |                |


| BBlloocckk 02 BBlloocckk 16 0DCX
BBlolocckk 120 BBlolocckk 144 |      |     | BBlloocckk 03 BBlloocckk 17 0DCX
BBlolocckk 121 BBlolocckk 145 |      |     |
| -------------------------------------------------------------- | ---- | --- | -------------------------------------------------------------- | ---- | --- |
| 2L                                                             | HQ 0 |     | 2L                                                             | HQ 0 |     |


| BBlloocckk 02  |
| -------------- |
| BBlolocckk 120 |


| BBlloocckk 03  | BBlloocckk 17  |
| -------------- | -------------- |
| BBlolocckk 121 | BBlolocckk 145 |



Figure7.NaiveBlock-firstmapping(illustratedwitheightqheads,  

128rowblocks,fourXCDs):WGsaredispatchedblock-by-block  

acrossallheadsinround-robinfashion,splittingACCsacross  

XCDs.Eachuniquecolorcorrespondstoauniqueattentionhead.  



3.2.2SwizzledBlock-first  

Figure9.NaiveHead-firstmapping(eightqheads,128rowblocks,  

TheSwizzledBlock-firstapproachretainstheblock-firstfourXCDs):WGsiteratethroughallblocksofeachheadsequen-  

iterationorderbutappliesaswizzlingtechniquetoco-locatetially,butround-robinXCDassignmentspreadseachheadacross  

allXCDs.AllblockscorrespondtothesameattentionheadinthisgroupedheadsinGQAwithinthesameXCD,preserving  

figure.locality(Figure8).However,thisonlymaintainslocal-  

itywhenthenumberofGQAgroupsmatchesthenumber  

ofXCDs.ForMHA,usedintheprefillstageofmodelsForillustration,Figures7–9showthesestrategiesfora  

likeDeepSeek-V3(Team,2024),thisstrategycausesmulti-simplifiedexamplewitheightqueryheads,128rowblocks  

pleACCsperXCDtobeservedsimultaneously,reducingperhead,andfourXCDs.  

6  


<!-- page 7 -->

3.3SwizzledHead-firstMapping  


| # Grid definition                           |
| ------------------------------------------- |
| grid = lambda META: (batch * num_q_heads *  |
| (seqlen_q/META["BLOCK_M"]))                 |
|                                             |
| # Swizzling                                 |
| wid = t1.program_id(0) # workgroup id       |
| wid_per_batch = wid // BATCH                |
| heads_per_xcd = NUM_Q_HEADS // NUM_XCD      |
| blocks_per_head = (SEQLEN_Q + BLOCK_M - 1)  |
| // BLOCK_M                                  |
| chunk_size = NUM_XCD * blocks_per_head      |
|                                             |
| # Calculate offsets                         |
| head_offset = ((wid_per_batch % NUM_XCD) *  |
| heads_per_xcd + wid_per_batch // (          |
| NUM_XCD * blocks_per_head))                 |
| block_offset = (wid_per_batch % chunk_size) |
| // NUM_XCD                                  |
| batch_offset = (wid // (blocks_per_head *   |
| NUM_Q_HEADS)) % BATCH                       |

1  

2TomaximizespatiallocalityonAMDGPUswithdisag-  

gregatedL2caches,weleverageakeyinsight:allblocks  

3  

withinthesameattentionheadandbatchshareKandV4  

tensors.Co-locatingtheseblockswithinanXCDimproves5  

datareuseandreducescross-devicetraffic.6  

7  

8  


| BBlloocckk 00 BBlloocckk 11 0DCX BBlloocckk 00 BBlloocckk 11 0DCX
BBlloocckk 22 BBlloocckk 43 BBlloocckk 22 BBlloocckk 43
HQ 0 2L HQ 2 2L
BBlloocckk 00 BBlloocckk 11 0DCX BBlloocckk 00 BBlloocckk 11 0DCX
BBlloocckk 22 BBlloocckk 43 BBlloocckk 22 BBlloocckk 43
HQ 4 2L HQ 6 2L |              |              |              |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------ | ------------ | ------------ |
| XCD0: HQ 0,1                                                                                                                                                                                                                                                                        | XCD1: HQ 2,3 | XCD2: HQ 4,5 | XCD3: HQ 6,7 |


| BBlloocckk 00 BBlloocckk 11 0DCX
BBlloocckk 22 BBlloocckk 43 | BBlloocckk 00 BBlloocckk 11 0DCX
BBlloocckk 22 BBlloocckk 43 |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| HQ 0 2L                                                      | HQ 2 2L                                                      |


| BBlloocckk 00 | BBlloocckk 11 |
| ------------- | ------------- |
| BBlloocckk 22 | BBlloocckk 43 |


| BBlloocckk 00 | BBlloocckk 11 |
| ------------- | ------------- |
| BBlloocckk 22 | BBlloocckk 43 |

9  

10  

11  

12  


| BBlloocckk 00 BBlloocckk 11 0DCX
BBlloocckk 22 BBlloocckk 43 |      |     | BBlloocckk 00 BBlloocckk 11 0DCX
BBlloocckk 22 BBlloocckk 43 |      |     |
| ------------------------------------------------------------ | ---- | --- | ------------------------------------------------------------ | ---- | --- |
| 2L                                                           | HQ 4 |     | 2L                                                           | HQ 6 |     |


| BBlloocckk 00 | BBlloocckk 11 |
| ------------- | ------------- |
| BBlloocckk 22 | BBlloocckk 43 |


| BBlloocckk 00 | BBlloocckk 11 |
| ------------- | ------------- |
| BBlloocckk 22 | BBlloocckk 43 |



13  

14  



Figure11.WorkgroupIDswizzlinglogicforSwizzledHead-first  

Mapping.TheswizzlingschememapsthelinearworkgroupIDFigure10.SwizzledHead-firstmapping(eightqheads,128row  

widblocks,fourXCDs):Head-firstiterationwithspatialswizzling()tobatch,head,andblockoffsets,ensuringattentionheads  

fromdifferentACCsaredistributedacrossXCDswhilesequenceconfineseachattentionheadtoasingleXCD,maximizingcache  

blockswithineachACCmaintainlocalityonthesameXCD.localityandensuringeachXCDservicesoneACCatatime.Each  

uniquecolorcorrespondstoauniqueattentionhead.  

Table1.AMDMI300XArchitectureSpecifications  

Weintroduceanovelmappingstrategy,SwizzledHead-  

ComponentSpecificationfirst,thatusesspatialswizzlingtoassignallblocksofan  

attentionheadtothesameXCDbeforemovingtothenextComputeArchitecture  

head.ThisapproachusesspatialswizzlingtoensurethatNumberofXCDs8  

workgroupswithinanACCaccessshareddatafromoneComputeUnitsperXCD38(304total)  

localcache,andXCDsserviceoneACCatatime.ThisStreamProcessorsperCU64  

strategyaddressesthelimitationsofexistingapproachesby  

MemoryHierarchy  

combininghead-firstiterationwithspatialswizzlingtoco-  

L1CacheperCU16KB  

locateallblocksofanattentionheadwithinasingleXCD  

L2CacheperXCD4MB(32MBtotal)  

(Figure10andcorrespondingswizzlingcodeinFigure11).  

HBM3Capacity192GB  

Remarkably,thisintuitivestrategyrequiresminimalcodeHBM3Bandwidth5.3TB/s  

modifications,asshowninFigure11,yetproveshighly  

effectiveatexploitingspatiallocality.Thehead-firstmap-  

pingconsistentlydeliversimprovedL2cachehitratesand  

memorycontrollers.Thisdisaggregatedmemoryhierarchy  

substantialperformanceimprovements,demonstratingthe  

createstheNUMAeffectsthatouroptimizationtargets.  

significantimpactofNUMA-awarekerneldesignonmulti-  

AllkernelsareimplementedinTriton(Tilletetal.,2019).diechiplet-basedGPUs.  

WeuseROCProfilerv3(AdvancedMicroDevices,Inc.,  

2025)tomeasureL2cachehitratesviahardwareperfor-4EVALUATION  

mancecounters,specificallybymonitoringtheaggregated  

L2cachehitrateacrossallXCDs.4.1ExperimentalSetup  

WeevaluateourspatiallocalityoptimizationsforFlashAt-  

4.2EvaluationMethodology  

tention2forwardpassonanAMDMI300XGPU,achiplet-  

basedarchitecturespecificallydesignedforAIworkloads.Ourexperimentalmethodologycomprisesthreecompo-  

Table1summarizesthekeyarchitecturalcharacteristicsrel-nents:(1)asensitivityanalysisacrossMHAmodelhyper-  

evanttoourNUMA-awareoptimizations.TheMI300Xparameterstounderstandhowsequencelength,headcount,  

featureseightXCDsorAcceleratorComplexDies,withandheaddimensionaffectthebenefitsofourapproach;(2)  

eachXCDcontainingitsowncomputeunits,L2cache,andperformancevalidationonGQAconfigurationstodemon-  

7  


<!-- page 8 -->

**Naive Block-firstSwizzled Block-firstNaive Head-firstSwizzled Head-first**  


|     |     |     |
| --- | --- | --- |
|     |     |     |
|     |     |     |
|     |     |     |
|     |     |     |



**1.00**  

**e**  

**cn**  

**a0.90m**  

**ro**  

**fr0.80e**  

**P**  

**0.70**  

**Batch124812481248124812481248124812481248124812481248124812481248**  

**N_CTX8K32K128K8K32K128K8K32K128K8K32K128K8K32K128K**  

**H_Q8163264128**  

Figure12.MHAPerformancerelativetoSwizzledHead-firstbaselineacrossvaryingbatchsizes(1-8)andsequencelengths(8K-128K).  

Block-firstapproachesshowsignificantperformancegapsthatwidenathigherheadcounts(HQ≥64),longersequences,andlarger  

batchsizes.  



**Naive Block-firstSwizzled Block-firstNaive Head-firstSwizzled Head-first**  

**100**  


|     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- |
|     |     |     |     |     |     |
|     |     |     |     |     |     |

**)80%**  

**(**  

**e 60**  

**ta40rtiH20**  

**0**  

**Batch124812481248124812481248124812481248**  

**N_CTX8K32K128K8K32K128K8K32K128K**  

**H_Q832128**  

Figure13.L2CachehitratesforMHAacrossvaryingbatchsizes(1-8)andsequencelengths(2K-128K).TheSwizzledHead-first  

approachconsistentlyachieveshighL2hitratesacrossallconfigurations,whereastheotherapproachesexhibitdrasticdropsinhitratesat  

highernumbersofheads,longersequencelengths,andhigherbatchsizes.  



strategeneralizationbeyondMHA;and(3)afocusedcase  

Table2.ExperimentalconfigurationforMulti-HeadAttention  

studyontheprefillstageoftheDeepSeekV3forwardpass,  

(MHA)SensitivityStudy  

representingareal-worlddeploymentscenariowith128  

ParameterValuesattentionheads.  

ContextLength(NCTX)8K,32K,128K  

WecompareallfourmappingstrategiesintroducedinSec-  

BatchSize1,2,4,8  

tion3:NaiveBlock-first(Figure7),SwizzledBlock-first  

NumberofHeads(HQ=HK)8,16,32,64,128  

(Figure8),NaiveHead-first(Figure9),andourproposed  

HeadDimension(DHEAD)128  

SwizzledHead-first(Figure10).Foreachexperiment,per-  

BlockSize(M×N)128×64  

formancemetricsarenormalizedtotheSwizzledHead-first  

approach,whichrepresentsthebest-performingconfigu-  

rationacrossourexperiments,enablingdirectassessment  

ofperformancedegradationinalternativeapproaches.Ad-  

fixedBLOCKM=128andBLOCKN=64(Table2).  

ditionally,fortheMHAsensitivitystudy,wemeasureL2  

cachehitratestovalidatetheimpactofimprovedcacheFigure12demonstratesthattheSwizzledHead-firstap-  

localityonperformance.proach,ourproposedmethod,servesastheperformance  

baseline.Forasmallernumberofheads,allapproaches  

performsimilarly.Theblock-firstapproachesshowsig-4.3Multi-HeadAttentionSensitivityStudy  

nificantperformancedegradationacrossmostconfigura-  

WeevaluatedthefourMHAmappingschemesacrossvary-tions,withthegapwideningasthenumberofheadsand  

ingconfigurationstoassesstheimpactofspatiallocalitysequencelengthincrease.Theperformancedisadvantage  

onperformanceandL2cacheutilization.OurexperimentsismostpronouncedatH≥64withlongersequencesQ  

sweepacrossnumberofattentionheads(8to128),sequence(N≥32K),whereBlock-firstapproachesachieveCTX  

lengths(2Kto128Ktokens),batchsizes(1to8),andusedonly64-70%theefficiencyofourSwizzledHead-first  

8  


<!-- page 9 -->

**Naive Block-firstSwizzled Block-firstNaive Head-firstSwizzled Head-first**  


|     |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
|     |     |     |     |     |     |     |     |     |
|     |     |     |     |     |     |     |     |     |
|     |     |     |     |     |     |     |     |     |
|     |     |     |     |     |     |     |     |     |
|     |     |     |     |     |     |     |     |     |

**1.00**  

**e**  

**c0.95n**  

**am0.90**  

**ro0.85fre**  

**P0.80**  

**0.75**  

**Batch124812481248124812481248124812481248**  

**N_CTX8K32K128K8K32K128K8K32K128K**  

**H_Q3264128**  

Figure14.PerformancecomparisonforGroupedQueryAttentionwitheightKVheads,normalizedtoSwizzledHead-firstbaseline.  

ResultsforHQ=32,64,128(correspondingtoLlama38B,70B,405Bmodels)acrosssequencelengths8K-128Kandbatchsizes1-8.  

Bothswizzledapproachesachievesimilarperformance,whileNaiveBlock-firstmappingshowssubstantialdegradationathigherattention  

headcountsandlongersequences.  



Thissubstantialdifferenceincacheutilizationdirectlytrans-  

Table3.ModelconfigurationsforLlamaModels(GQA)and  

latestotheobservedperformancebenefits,asreducedcache  

DeepSeek-V3(MHA)  

missesminimizeexpensiveoff-chipmemoryaccesses.The  

ModelAttn.TypeHQHKDHEADSwizzledHead-firststrategy’sabilitytomaintainhighcache  

utilizationunderextremeconditionsdirectlytranslatestoLlama-38BGQA328128  

sustainedcomputationalthroughput,makingitessentialforLlama-370BGQA648128  

scalingattentionmechanismstolargeheadcountsandlongLlama-3405BGQA1288128  

contextlengths.  

DeepSeek-v3MHA12812856  

4.4GroupedQueryAttentionSensitivityStudy  

WeevaluatedthefourmappingschemesonGQAconfigura-  

method.AtH=128withN=128K,theSwiz-QCTXtionswithafixedeightKVheadsandvaryingqueryheads  

zledHead-firstapproachachievesupto50%higherperfor-(HQ=32,64,128),correspondingtotheLlama3model  

mancethanBlock-firstmappings.Interestingly,theNaivefamilysizes(8B,70B,and405Bparameters,respectively).  

Head-firstmapping’sperformancecloselymatchesourswiz-AsmultiplequeryheadssharethesameKVheads,GQA  

zledapproachformostcases,exceptatveryhighsequencecreatesdifferentmemoryaccesspatternsandcachepressure  

lengths,whereitdropsto∼90%relativeefficiency.ThischaracteristicscomparedtoMHA.  

observationalignswithourtheorythattheNaiveHead-first  

Figure14demonstratesthattheSwizzledHead-firstandapproachstillbenefitsfromcachelocalitybyhavingtherel-  

SwizzledBlock-firstapproachesmaintainrobustperfor-evantACC’sdatainthecaches,evenifit’sreplicatedacross  

manceacrossallGQAconfigurations.AtHQ=32,allallcaches,exceptatveryhighsequencelengths,where  

mappingschemesachievesimilarperformance.TheNaivecachesplittingwithdifferentACC’sdataandredundant  

Block-firstmethodexhibitssignificantdegradationathighermemoryfetchesdegradeperformance.  

queryheads,sequencelengths,andbatchsizes,sincetheL2  

TheL2cachehitrateanalysisinFigure13revealstheunder-cacheineachXCDissplitacrossdifferentACCs’data.  

lyingcauseoftheseperformancedifferences.Withfewer  

TheSwizzledBlock-firstapproachperformswellwithGQA,headsandshortersequences,allapproachesmaintainhigh  

astheeightKVheadsmatchthenumberofXCDsintheL2hitrates(∼90%).However,cachebehaviordiverges  

MI300Xarchitecture,avoidinganycachesplits.Inthecasedramaticallyasworkloadcomplexityincreases.AtHQ=  

ofSwizzledHead-first,eachACC,whichisnowagroup128withNCTX=128K,theSwizzledHead-firstmapping  

ofallqueryheadssharingtheKVdata,ismappedtoonesustains90-96%hitratesacrossallbatchsizes,thehit-rate  

XCD,achievingsimilarlystrongperformance.TheNaiveofNaiveHead-firstfallsto40-60%,whileblock-firstap-  

Head-firstapproachshowsinstabilityathighersequenceproachescollapsetoapproximately1%hitrates.Eventhe  

lengthsandbatches,withperformancedippingtoaboutSwizzledBlock-firstmethod,whichisdeployedincurrent  

90-95%oftheSwizzledHead-firstapproachinsomecases,GPUkernels,almostalwaysmissesincacheattheseex-  

indicatingnon-idealcacheutilizationandhigherredundanttremeconfigurations,despiteperformingbetteratmoderate  

memoryaccesses.Theseresultsconfirmthatspatiallocalityscales.Thiscatastrophiccachedegradationdirectlyexplains  

optimizationsremaineffectiveforGQAworkloads.the45-50%performancegapobservedinFigure12.  

9  


<!-- page 10 -->

**Naive Block-firstSwizzled Block-firstNaive Head-firstSwizzled Head-first**  


|     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- |
|     |     |     |     |     |     |     |
|     |     |     |     |     |     |     |
|     |     |     |     |     |     |     |
|     |     |     |     |     |     |     |

**1.00**  

**ecn0.90a**  

**m**  

**ro0.80fre**  

**P0.70**  

**Batch1248124812481248124812481248**  

**N_CTX2K4K8K16K32K64K128K**  

Figure15.DeepSeekV3prefillperformancewith128attentionheads,normalizedtoSwizzledHead-firstbaseline,acrosssequence  

lengths2K-128Kandbatchsizes1-8.Block-firstapproachesshowsignificantperformancedegradationatlongersequences,withNaive  

Block-firstdroppingbelow0.65×relativeefficiencyat128Ktokens.  



TheseresultsconfirmthattheSwizzledHead-firstoptimiza-  

**Naive Block-firstNaive Head-first**  

**Swizzled Block-firstSwizzled Head-first**tionprovidessubstantialandscalableperformancebenefits  

forDeepSeekV3’shigh-head-countMHAconfiguration.  


|     |     |     |     |     |
| --- | --- | --- | --- | --- |
|     |     |     |     |     |
|     |     |     |     |     |
|     |     |     |     |     |
|     |     |     |     |     |
|     |     |     |     |     |

**1.00e**  

**cn0.98a**4.6FlashAttention2BackwardPass  

**m**  

**r0.96**  

**o**WeevaluatedthefourmappingschemesfortheFlashAt-**fr0.94e**tention2backwardpasskernelinAMD’sAITER(AMD,**P0.92**  

2025a)repositorywithHQ=128attentionheadsacross  

**Batch121212**varyingcontextlengths(8K,32K,128K)andbatchsizes  

**N_CTX8K32K128K**  

(1,2).Figure16showsthattheSwizzledHead-firstap-  

proachconsistentlyoutperformsothermappingschemes,  

Figure16.BackwardpassspeedupcomparisonofmappingwiththeperformanceoftheSwizzledBlock-firstapproach  

schemeswith128queryheadsversustheNaiveBlock-firstap-  

degradingtoabout0.94×andtheperformanceoftheNaive  

proach.Resultsshownacrosscontextlengthsfrom8Kto128K  

Block-firstandNaiveHead-firstapproachesdegradingto  

tokensandbatchsizes1to8.TheSwizzledHead-firstapproach  

0.91×at128Ktokens.Giventheadditionalcomplexityanddemonstratesincreasingspeedupwithlongersequences,achieving  

scalaroperationsrequiredintheFlashAttention2backward1.10×at128Ktokens.  

pass,wesuspectthatfurthergainsmaybeconstrainedby  

emergingbottlenecksintroducedbytheSwizzledHead-first  

optimization.Thisbehavioralignswithourobservations  

4.5CaseStudy:OptimizingDeepSeekV3Prefillacrossotherkernels,whereperformanceimprovementsof  

upto50%wereachieved.WeleavefurtheroptimizationtoDeepSeekV3’sprefillphaseusesMHAwith128query  

futurework.headsand128key-valueheads,makingitanidealcan-  

didateforevaluatingourSwizzledHead-firstoptimization,  

asthenumberofattentionheads(128)significantlyexceeds5CONCLUSION  

thenumberofXCDs(8)availableontheMI300XGPU.  

Thisworkdemonstratesthatspatiallocalityinattention  

AsshowninFigure15,theSwizzledHead-firstapproachworkloadscanbestrategicallyexploitedtomitigateNUMA  

achievessuperiorperformanceacrossmostconfigurations,effectsinchiplet-basedGPUswithdisaggregatedmemory.  

particularlyatlongersequencelengths.Atextremese-Weidentifyinherentspatiallocalitypatternsintheattention  

quencelengths,theNaiveBlock-firstmethodshowsthemechanismandintroducetheSwizzledHead-firstMapping  

worstperformance,degradingtounder65%relativeperfor-strategy,whichalignscomputationwiththeNUMAdo-  

manceat128Ktokens.TheSwizzledBlock-firstmethodmainsofmulti-dieGPUarchitectures.OnAMD’sMI300X  

alsodegradessubstantiallyto76%oftheSwizzledHead-architecture,ourapproachachievesupto50%performance  

firstmethodat128Ktokenswithabatchsizeofeight.Theimprovementoverstate-of-the-artattentionalgorithmsus-  

NaiveHead-firstapproachalsoshowsinstabilityatextremeingconventionalschedulingtechniquesinmulti-headat-  

lengths,withperformancevaryingsignificantly.ThesmallertentionwhilemaintainingconsistentlyhighL2cachehit  

headdimensioninthisconfiguration(DHEAD=56)reducesratesof80-97%.Theseresults,achievedwithminimalcode  

overallarithmeticintensity,therebyloweringabsoluteper-changes,demonstratethatsignificantperformancegainscan  

formanceacrossallmethods.  

10  


<!-- page 11 -->

beobtainedviaarchitecture-awarekerneldesign.Ourfind-NVIDIA.NVIDIABlackwellArchitecture.https:  

ingsunderscorethenecessityofhardware-awarealgorithm//www.nvidia.com/en-us/data-center/  

designaschiplet-basedarchitecturesbecomeincreasinglytechnologies/blackwell-architecture/,  

prevalentinAIaccelerators.2024a.  

https://www.NVIDIA.NVIDIARubinPlatform.  

REFERENCESnvidia.com/gtc/keynote/,2024b.Announced  

atNVIDIAGTC2024.AdvancedMicroDevices,Inc.ROCmProfiler(rocprofv3)  

https://rocm.docs.amd.com/Documentation.  

NVIDIA.NVIDIARubinUltraPlatform.https://www.  

projects/rocprofiler-sdk/en/latest/  

nvidia.com/gtc/keynote/,2024c.Announced  

how-to/using-rocprofv3.html,2025.Ac-  

atNVIDIAGTC2024.  

cessed:14-October-2025.  

Sanovar,R.,Bharadwaj,S.,Amant,R.S.,Ru¨hle,V.,and  

Ainslie,J.,Lee-Thorp,J.,deJong,M.,Zemlyanskiy,Y.,Rajmohan,S.LeanAttention:Hardware-awarescalable  

Lebro´n,F.,andSanghai,S.Gqa:Traininggeneralizedattentionmechanismforthedecode-phaseoftransform-  

multi-querytransformermodelsfrommulti-headcheck-ers,2025.URLhttps://arxiv.org/abs/2405.  

points,2023.URLhttps://arxiv.org/abs/10480.  

2305.13245.  

Shah,J.,Bikshandi,G.,Zhang,Y.,Thakkar,V.,Ramani,P.,  

AMD.AMDInstinctMI200SeriesAccelerators.andDao,T.FlashAttention-3:fastandaccurateattention  

https://www.amd.com/en/products/withasynchronyandlow-precision.InProceedingsof  

accelerators/instinct/mi200.html,2021.the38thInternationalConferenceonNeuralInformation  

ProcessingSystems,NIPS’24,RedHook,NY,USA,  

AMD.AMDInstinctMI300SeriesAccelerators.2025.CurranAssociatesInc.ISBN9798331314385.  

https://www.amd.com/en/products/  

accelerators/instinct/mi300.html,2023.Team,D.A.DeepSeek-V3:Amixture-of-expertslan-  

guagemodelwithmulti-headlatentattention.arXiv  

AMD.Aitensorengineforrocm,2025a.URLhttps:preprintarXiv:2412.19437,2024.URLhttps://  

//github.com/ROCm/aiter.Accessed:2025-07-arxiv.org/abs/2412.19437.  

19.  

Tillet,P.,Kung,H.T.,andCox,D.Triton:aninter-  

AMD.hipBLASlt,2025b.URLhttps:mediatelanguageandcompilerfortiledneuralnet-  

//github.com/ROCm/rocm-libraries/workcomputations.InProceedingsofthe3rdACM  

tree/develop/projects/hipblaslt.Ac-SIGPLANInternationalWorkshoponMachineLearn-  

cessed:2025-07-19.ingandProgrammingLanguages,MAPL2019,pp.  

10–19,NewYork,NY,USA,2019.AssociationforCom-  

AMD.Tensile,2025c.URLhttps://github.putingMachinery.ISBN9781450367196.doi:10.  

com/ROCm/rocm-libraries/tree/develop/https://doi.org/1145/3315508.3329973.URL  

shared/tensile.Accessed:2025-07-19.10.1145/3315508.3329973.  

TritonDevelopers.triton.language.swizzle2d,2025.Dao,T.FlashAttention-2:Fasterattentionwithbetterparal-  

URLhttps://triton-lang.org/main/lelismandworkpartitioning.InInternationalConference  

python-api/generated/triton.language.onLearningRepresentations(ICLR),2024.  

swizzle2d.html.Tritondocumentation(version:  

Dao,T.,Fu,D.Y.,Ermon,S.,Rudra,A.,andRe´,C.FlashAt-main).  

tention:Fastandmemory-efficientexactattentionwith  

Vaswani,A.,Shazeer,N.,Parmar,N.,Uszkoreit,J.,Jones,IO-awareness.InAdvancesinNeuralInformationPro-  

L.,Gomez,A.N.,Kaiser,L.,andPolosukhin,I.AttentioncessingSystems(NeurIPS),2022.  

isallyouneed.InProceedingsofthe31stInternational  

NVIDIA.NVIDIAA100TensorCoreGPUAr-ConferenceonNeuralInformationProcessingSystems,  

chitecture.https://www.nvidia.com/en-us/NIPS’17,pp.6000–6010,RedHook,NY,USA,2017.  

data-center/a100/,2020.CurranAssociatesInc.ISBN9781510860964.  

NVIDIA.NVIDIAH100TensorCoreGPUAr-  

chitecture.https://www.nvidia.com/en-us/  

data-center/h100/,2022.  

11  
