#define INPUT_DIMS 8
#define PATCH_DIM 6

static float input[INPUT_DIMS * PATCH_DIM * PATCH_DIM] = { 1.1889175176620483, 0.0, 0.686677873134613, 0.28590407967567444, 0.6756686568260193, 0.0, 1.0730130672454834, 1.5199142694473267, 1.2245371341705322, 0.0, 0.9322224259376526, 0.24574483931064606, 0.9178060293197632, 0.0, 1.2070866823196411, 1.2803878784179688, 1.2177472114562988, 0.0, 0.9349721074104309, 0.23399050533771515, 0.9389560222625732, 0.0, 1.20634925365448, 1.2531741857528687, 1.2091892957687378, 0.0, 0.9317412376403809, 0.2136395275592804, 0.9431822299957275, 0.0, 1.2010629177093506, 1.224714994430542, 1.2066428661346436, 0.0, 0.9345977306365967, 0.19556978344917297, 0.933709442615509, 0.0, 1.2079591751098633, 1.2254459857940674, 1.2132079601287842, 0.0, 0.9328851699829102, 0.17955024540424347, 0.9042797088623047, 0.0, 1.2169342041015625, 1.2241672277450562, 1.1965123414993286, 0.0, 0.9510436058044434, 0.22919078171253204, 0.9852019548416138, 0.0, 1.187132716178894, 1.0319616794586182, 1.2993309497833252, 0.0, 1.3039311170578003, 0.06124335899949074, 1.41031014919281, 0.0, 1.2472972869873047, 0.16132937371730804, 1.2937241792678833, 0.0, 1.3021215200424194, 0.06260563433170319, 1.4223699569702148, 0.0, 1.2345714569091797, 0.16947343945503235, 1.2845354080200195, 0.0, 1.3057010173797607, 0.058147184550762177, 1.4295300245285034, 0.0, 1.2378731966018677, 0.1907886415719986, 1.283066987991333, 0.0, 1.3140215873718262, 0.03941822052001953, 1.4319969415664673, 0.0, 1.2510833740234375, 0.18278124928474426, 1.293532371520996, 0.0, 1.311673641204834, 0.03419610112905502, 1.3864810466766357, 0.0, 1.2727792263031006, 0.2050451636314392, 1.1904404163360596, 0.0, 0.9395459294319153, 0.2246854305267334, 0.9816352128982544, 0.0, 1.1698871850967407, 1.0149723291397095, 1.3004311323165894, 0.0, 1.2942228317260742, 0.06068005785346031, 1.3776077032089233, 0.0, 1.2450120449066162, 0.18701013922691345, 1.2921385765075684, 0.0, 1.293941617012024, 0.06267154216766357, 1.4103981256484985, 0.0, 1.229309320449829, 0.18862979114055634, 1.2881702184677124, 0.0, 1.2982115745544434, 0.04518850892782211, 1.4149149656295776, 0.0, 1.2361881732940674, 0.16846024990081787, 1.2842456102371216, 0.0, 1.2991424798965454, 0.032028645277023315, 1.4036457538604736, 0.0, 1.240525722503662, 0.1888081133365631, 1.28203547000885, 0.0, 1.2897261381149292, 0.028658922761678696, 1.3826022148132324, 0.0, 1.24242103099823, 0.18638011813163757, 1.1907427310943604, 0.0, 0.9360589981079102, 0.24961833655834198, 1.0011098384857178, 0.0, 1.1607639789581299, 1.030193567276001, 1.296494483947754, 0.0, 1.296299934387207, 0.08141043037176132, 1.4052566289901733, 0.0, 1.2397780418395996, 0.19067169725894928, 1.2931671142578125, 0.0, 1.2930649518966675, 0.06282022595405579, 1.3982316255569458, 0.0, 1.2368018627166748, 0.18314552307128906, 1.293848991394043, 0.0, 1.2926746606826782, 0.04599253088235855, 1.3803800344467163, 0.0, 1.2454861402511597, 0.17322129011154175, 1.2910401821136475, 0.0, 1.3079243898391724, 0.0510125495493412, 1.4048796892166138, 0.0, 1.2561426162719727, 0.19181905686855316, 1.2922446727752686, 0.0, 1.2961087226867676, 0.05065854266285896, 1.3945356607437134, 0.0, 1.2462090253829956, 0.20390088856220245, 1.1844453811645508, 0.0, 0.9524935483932495, 0.24208222329616547, 1.0300124883651733, 0.0, 1.170817255973816, 1.0158674716949463, 1.2910248041152954, 0.0, 1.3042423725128174, 0.07530564814805984, 1.4385987520217896, 0.0, 1.2299007177352905, 0.1766124963760376, 1.2868092060089111, 0.0, 1.287881851196289, 0.05629950016736984, 1.4077354669570923, 0.0, 1.216823697090149, 0.1919325590133667, 1.2880289554595947, 0.0, 1.2860686779022217, 0.06910263001918793, 1.390699028968811, 0.0, 1.2278554439544678, 0.21591787040233612, 1.297263264656067, 0.0, 1.3117581605911255, 0.060278668999671936, 1.4131267070770264, 0.0, 1.2561383247375488, 0.18795517086982727, 1.2967503070831299, 0.0, 1.3005088567733765, 0.05568600445985794, 1.3861087560653687, 0.0, 1.2502738237380981, 0.21080316603183746, 1.1885993480682373, 0.0, 0.9490041732788086, 0.24165378510951996, 1.0236011743545532, 0.0, 1.1671867370605469, 1.015089750289917, 1.2957342863082886, 0.0, 1.3018475770950317, 0.08859658241271973, 1.4310593605041504, 0.0, 1.2304565906524658, 0.2020004540681839, 1.299179196357727, 0.0, 1.2856945991516113, 0.07151154428720474, 1.3883033990859985, 0.0, 1.2264938354492188, 0.1973874419927597, 1.296501636505127, 0.0, 1.2827900648117065, 0.06527338176965714, 1.3756159543991089, 0.0, 1.2313485145568848, 0.20322437584400177, 1.2951271533966064, 0.0, 1.2936677932739258, 0.04819206893444061, 1.3807140588760376, 0.0, 1.2414236068725586, 0.17494681477546692, 1.292358636856079, 0.0, 1.2943769693374634, 0.044796135276556015, 1.3858633041381836, 0.0, 1.2440060377120972, 0.16450615227222443 };
