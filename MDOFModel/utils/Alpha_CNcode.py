########################################################
# Design strength based on CN code
# 
# Dependancy: 
# -
########################################################

def Alpha_CNcode(T,Tg,alphaMax,kesi=0.05):

    gamma = 0.9 + (0.05-kesi) / (0.3+6*kesi)
    eta1 = 0.02 + (0.05-kesi) / (4+32*kesi)
    if eta1<0:
        eta1 = 0
    eta2 = 1 + (0.05-kesi) / (0.08 + 1.6*kesi)
    if eta2<0.55:
        eta2 = 0.55

    if ((T>=0) and (T<0.1)):
        alpha = (eta2 * alphaMax - 0.45*alphaMax)/0.1*T + 0.45*alphaMax
    elif (T>=0.1 and T<Tg):
        alpha = eta2 * alphaMax
    elif (T>=Tg and (T<5*Tg)):
        alpha = (Tg/T)**gamma*eta2*alphaMax
    elif (T>=5*Tg and T<6):
        alpha = (eta2*0.2**gamma-eta1*(T-5*Tg))*alphaMax
    elif (T>=6):
        alpha = (eta2*0.2**gamma-eta1*(6-5*Tg))*alphaMax

    return alpha


# Tg
def Tg_CNcode(EQgroup,SiteClass):

    matrix = {
        '1': {'1_0': 0.20, '1_1': 0.25, '2': 0.35, '3': 0.45, '4': 0.65},
        '2': {'1_0': 0.25, '1_1': 0.30, '2': 0.40, '3': 0.55, '4': 0.75},
        '3': {'1_0': 0.30, '1_1': 0.35, '2': 0.45, '3': 0.65, '4': 0.90}
    }

    if EQgroup in matrix and SiteClass in matrix[EQgroup]:
        Tg = matrix[EQgroup][SiteClass]
    else:
        Tg = None

    return Tg

# alphaMax
# EQlevel - 'minor', 'medium', 'major'
# SeismicDesignLevel - '6', '7', '7.5', '8', '8.5', '9'
def alphaMax_CNcode(EQlevel,SeismicDesignLevel):
    matrix = {
        'minor':  {'6': 0.04, '7': 0.08, '7.5': 0.12, '8': 0.16, '8.': 0.24, '9': 0.32},
        'medium': {'6': 0.12, '7': 0.24, '7.5': 0.36, '8': 0.48, '8.': 0.72, '9': 0.96},
        'major':  {'6': 0.28, '7': 0.50, '7.5': 0.72, '8': 0.90, '8.': 1.20, '9': 1.40}
    }
    if EQlevel in matrix and SeismicDesignLevel in matrix[EQlevel]:
        alphaMax = matrix[EQlevel][SeismicDesignLevel]
    else:
        alphaMax = None
    return alphaMax

# convert Chinese seismic design level to Hazus seismic design level
def Concert_CN2Hazus_SeismicDesignLevel(SDL_CN: str):
    alphaMax = alphaMax_CNcode('medium',SDL_CN)
    if alphaMax > (0.4+0.2)/2:
        # UBC Zone 4 (0.4 g)
        SDL_Hazus = 'high-code'
    elif (alphaMax > (0.2+0.075)/2) and (alphaMax<= (0.4+0.2)/2):
        # UBC Zone 2B (0.2 g)
        SDL_Hazus = 'moderate-code'
    elif alphaMax<= (0.2+0.075)/2:
        SDL_Hazus = 'low-code'
    return SDL_Hazus
