from scipy.stats import gaussian_kde
from scipy.optimize import minimize
import numpy as np

# Step 1: Convert profits to PMF
def profits_to_pmf(profits):
    values, counts = np.unique(profits, return_counts=True)
    probabilities = counts / sum(counts)
    return values, probabilities

# Step 2: Perform KDE
def calculate_kde(profits):
    kde = gaussian_kde(profits)
    return kde

# Step 3: Perform MLE using KDE
def mle_with_kde(kde, profits):
    def likelihood(params):
        loc, scale = params
        # 使用一个小的正数来避免除以零
        scale = max(scale, 1e-10)
        # 计算调整后的数据
        adjusted_data = (profits - loc) / scale
        # 使用kde评估这些点的密度
        kde_values = kde(adjusted_data)
        # 避免取对数0或负数
        kde_values[kde_values <= 0] = 1e-10
        # 计算似然
        log_likelihood = np.sum(np.log(kde_values) - np.log(scale))
        return -log_likelihood  # 最小化负似然

    initial_params = [np.mean(profits), np.std(profits)]
    result = minimize(likelihood, initial_params, bounds=((None, None), (1e-10, None)))
    
    if result.success:
        return result.x
    else:
        return None, result.message


# Usage
a = '''-0.09156212714364210 0.0312516868105146 -0.0238229240620208 -0.04786127242790890 -0.07371925168988470 -0.07900318740194650 -0.09331611918841450 0.04166336586620890 0.04111544400829810 0.016484528050559400 -0.007867522295084920 -0.03627658938177200 -0.08286603570744320 -0.10870875703089100 -0.15942588097034700 0.5509769575305170 0.546547825957806 0.530319490545271 0.5144522052366430 0.5014156220556170 0.45312403740182400 0.4093544139196330 0.3952326252657880 0.3808263073770580 0.35658471473019700 0.3461435635713870 0.33881287509803700 0.29996626058785400 0.25459703559478800 0.21615708280303400 0.16222239269460500 0.05507735947256730 -0.05415396570542000 -0.06438246145751050 0.07528418150707660 0.058441162734634000 0.010716390735559600 0.0021395704781548300 -0.07395551939513790 -0.09958058632559050 -0.051822224500955400 -0.08440403517186530 -0.08694024523660030 0.0491757107448636 0.046590817269352300 0.03532004520020290 0.02103159796837310 0.002249738841886280 0.00030999319503144400 -0.018342296322716800 -0.023715777452753800 -0.03337678108081750 0.503173113189082 0.34801038910405000 0.3310678961003820 0.32530207478598400 0.3224256219779610 0.24809233079534800 0.24024866846016600 0.23745562129594000 0.2145367268173380 0.16625077588073000 0.05864696853217440 0.00928374327667214 -0.07474757213416320 0.013809448805490100 -0.0035037762552824200 -0.0060873518462842800 -0.012655941396287600 -0.031125892434167400 0.1086676072241680 0.10703173612439800 0.0804291792973213 0.08033552856217300 0.04762346190217960 0.03272221957364340 0.029297708695009000 -0.00504831698578001 -0.026782636769509300 -0.04288178368345620 0.09592643277490630 0.08578535877921080 0.08571890148973990 0.07759841415692300 0.04893207180536590 0.04570862447106850 0.0465692746847608 0.04648206572225420 0.03969002185307690 0.03201140375492910 0.03039741779994930 0.01959084272906360 -0.021399123470306800 -0.056144526998486300 -0.11478314588274400 -0.12676272227519500 -0.16284653523052600 -0.18254018401099300 -0.20093060736981300 -0.21837490082886700 -0.2942784518014020 -0.042074951981595200 -0.059027396566771600 -0.061130824390667500 0.11545736446371900 0.07991640092461690 0.07752900831140770 0.02658875314751970 0.028606062571287400 0.019624370075694000 -0.016975206924806600 -0.03743402211004970 -0.05756808140145210 0.8566571827417140 0.8305459482162150 0.8268311252511780 0.7997600610886030 0.7776679894601700 0.7556322521779250 0.707855796498974 0.702519830779577 0.6691461829057610 0.5837173277638040 0.548040167911813 0.5096798352344610 0.4605449437673040 0.44559053122099600 0.4248513245080240 0.40928150326354100 0.30548344767807000 0.2812321263494620 0.26406864108041800 0.21928629291302100 0.18612072823466400 0.13419238173906600 0.08876496646193630 0.02689856522887110 -0.030525178333722700 -0.08001263596381640 -0.09936822755361950 -0.11848174178898000 -0.16534488709212900 -0.21727915819967900 -0.23670613387103500 -0.2395001186768470 0.2702055466891980 0.2556556098188540 0.244245869754512 0.10675123651315500 0.06977112622421160 0.0540767504676416 -0.01239946816049230 -0.07079205942183840 -0.11331154494367300 -0.16677324906576200 -0.17769109338564900 -0.17769301653568000 -0.21114220278631900 -0.2331883819178520 -0.26442458075927300 0.19832815132092600 0.15776092130636600 0.15460860151380700 0.14496688231535300 0.01199618732405260 -0.02016832453024780 -0.05247649371253110 -0.07788846885789720 -0.08397870645345630 0.1050107826535310 0.04906172005951430 -0.0030680780461183100 -0.046638885821627700 -0.05687847387913290 -0.08745262699688140 -0.08729240838880020 -0.09263737431555240 1.0637421574246200 1.0161217015032900 0.9751713641763650 0.9687797893700120 0.9665510202084460 0.8440857090361770 0.732944384645374 0.6499067684950750 0.6083258085628510 0.589212612853607 0.47634364336497200 0.4624363058543120 0.43236143415286700 0.38697235609394900 0.35932222357479200 0.13610191827356600 -0.08960097646447220 -0.46475662072718000 -0.33461314319465800 -0.3366325441171550 -0.3347693178576540 -0.046787499115071100 -0.04777761963241170 -0.12303352181625900 -0.12253330377322800 -0.13588513466451900 0.044404551527734700 0.03550366188819120 0.023134346671376000 -0.00989520498888885 -0.08825021804778460 -0.13472797978314000 -0.013407305150948300 -0.022162467410602700 -0.03525775013512170 -0.034934334934035400 -0.06904936136070010 0.28033280465836300 0.27617886690544500 0.26037955259538900 0.07363552059483230 0.053907072353273700 0.04020480663913500 0.034862629867508800 0.007588856952692290 -0.009569470244463860 -0.016109666285348800 -0.01810060747281880 -0.029645975992724900 -0.05962694958928320 0.3317930514216400 0.3176631641337270 0.284349316009749 0.24410694578687100 0.2043550225795230 0.06661708506926760 -0.01594545825515700 -0.03829968508052840 -0.06237316881584170 -0.09066634022437830 -0.10395968554860400 0.26573046802359900 0.24496332904809900 0.1987488978122570 0.1748864393249370 0.09974098888927600 0.046200185525541600 0.031061759710470300 0.01668710931233710 -0.04989569785945100 -0.14029553119704500 -0.15183633992078200 -0.20452748250869500 -0.2314222792673690 0.16371470380872100 0.10764530921299300 0.1076106542419070 0.10388050516401500 0.024382718546610100 0.012693516164394600 0.011081464602738800 -0.03387037911484510 -0.0491973529688734 -0.06002312258878420 -0.07047841875024660 -0.07684420372921150 -0.08347259854401470 -0.10627488670571500 -0.11388099599822800 -0.12431578567317800 0.2377126287651870 0.21864794759200400 0.21548366643236200 0.1981870721600540 0.18012256657055800 0.17864986280815200 0.04588835261828320 0.03746992058806910 0.024240382249947500 0.021530986087882100 -0.006533760432487100 -0.07395024349838750 -0.14435809242249600 0.1642725996359630 0.13782774888483700 0.022960617517827000 0.015369477427607200 0.014421031586238100 -0.004604844191028910 -0.04152927809609610 -0.043167284399664900 -0.051442009379555600 -0.05255922383308380 -0.08010710395730010 0.5575431989707720 0.5443428668168160 0.5234706667325460 0.4978864102872330 0.47687583349141200 0.374366221394397 0.35636352880404100 0.3422376061323980 0.2885497953132980 0.2766955936800360 0.2611842243803390 0.2448328581668070 0.12929661271939100 0.12055288770280100 0.08095721654049620 0.07821682311369310 0.05444451262435140 -0.002893460457597750 -0.011779780898999000 -0.012513172290157800 -0.05516154933434430 -0.05622162924881940 -0.0790537054673568 0.8530895993738860 0.7074588809383070 0.5952759308468210 0.5727897839877250 0.5247461016162760 0.4743733467324420 0.37609670432515400 0.34345279652768100 0.32978854681570900 0.26148225607019200 0.2548500330942470 0.23851342913213800 0.13239353443811100 0.1088717327730170 0.06966389644712100 -0.01223712915353050 -0.07614263766519470 -0.10800817195902800 0.12018457587272700 0.11332001654415400 0.0805956491922919 0.06241061545886060 0.05138342697004130 -0.007772328546183480 -0.07428552430564420 -0.07727771187915260 -0.10076041169614300 -0.025449418699198000 -0.07957106059351190 -0.11293494355328200 0.2760784050788960 0.20413431168138900 0.1893137787655290 0.1883620525311450 0.127282207322309 0.1105308772702770 0.0671395121819165 0.027793264356223800 -0.005093924040867840 -0.035678798252923200 -0.040256449922729400 -0.04883450771643940 -0.08116556432293410 -0.09672029070196200 -0.01383258152029130 -0.05134444168491060 -0.11420925770531200 -0.12040029256000100 -0.15106944904179100 -0.17316209859801900 -0.19210416645747700 -0.07569911157615610 -0.08093251170952590 -0.08568423224443210 -0.02397669078597240 -0.07226513107554690 -0.07775242379436500 -0.0846796619178587 -0.04545751237021890 -0.08321542429794520 -0.10059443983570700 -0.10446770213526400 -0.10479879702290300 -0.12112317563879600 -0.12208983864434800 -0.03874485320250070 -0.040830992326857400 -0.05565750003631880 -0.061878768835235500 -0.04013564620134290 -0.041837191131946500 0.206971641886311 0.15891305970459600 0.09828387630899770 0.04242346148650930 0.032659298896525700 -0.035718595073236800 -0.039975634343674000 -0.04632456863387210 -0.05372519866617810 -0.08022102061811930 -0.14634072680701700 -0.15530455397898900 -0.15786642154358400 -0.16441212900647900 0.1379226153490030 0.03871573231252180 0.016731911031515800 0.012503069519625800 0.006590844647217380 0.005666656919213640 0.0013155859136033700 -0.03854120481813790 -0.05692030551699670 -0.06243994317467290 -0.06493637757243310 -0.0055890133216304200 -0.06143689997308020 -0.08322256689332060 -0.09640851081263810 -0.10553365919092600 0.4755667404217380 0.44735616405066600 0.4367418384144150 0.3907322097223330 0.3797958104755420 0.369631549677063 0.24803499690962100 0.21717129241164600 0.1964203574428730 0.1954691897477530 0.19113911046456900 0.16477978797866200 0.16550660427292800 0.15850197117724600 0.12477120214882400 0.10644749548174500 0.08980082204241090 0.0670045213565793 0.045840528289832200 0.03259253715604830 -0.016782434933050300 -0.0634843034480087 -0.06560399265780330'''.split()
profits = list(map(float, a))  # Replace this with your actual profits list
kde = calculate_kde(profits)
parameters = mle_with_kde(kde, profits)
print(parameters)