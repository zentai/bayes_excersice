import sys, os
import click
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
from collections import Counter
import seaborn as sns

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from huobi.client.market import MarketClient
from huobi.client.generic import GenericClient
from huobi.utils import *
from huobi.constant import *
from .utils import pandas_util
from .hunterverse.interface import Symbol
from config import config
from .cloud_story import DEBUG_COL, params
from .sensor.market_sensor import HuobiMarketSensor, YahooMarketSensor
from .strategy.turtle_trading import TurtleScout
from .hunterverse.interface import StrategyParam

DATA_DIR, SRC_DIR, REPORTS_DIR = config.data_dir, config.src_dir, config.reports_dir

from .story import entry

stocks_singapore = {
    "VNM.SI": "",
    "GRU.SI": "",
    "LCS.SI": "",
    "SQU.SI": "",
    "SHD.SI": "",
    "YLU.SI": "",
    "TATD.SI": "",
    "S45U.SI": "",
    "SSS.SI": "",
    "CXS.SI": "",
    "ICU.SI": "",
    "YLD.SI": "",
    "ICM.SI": "",
    "ESG.SI": "",
    "JJJ.SI": "",
    "VND.SI": "",
    "MCN.SI": "",
    "TCPD.SI": "",
    "LSS.SI": "",
    "GRO.SI": "",
    "LCU.SI": "",
    "SCY.SI": "",
    "TPED.SI": "",
    "SQQ.SI": "",
    "YWTR.SI": "",
    "W4VR.SI": "",
    "AYV.SI": "",
    "Z4D.SI": "",
    "BJD.SI": "",
    "YYB.SI": "",
    "KUH.SI": "",
    "BKK.SI": "",
    "BFK.SI": "",
    "UUK.SI": "",
    "5OR.SI": "",
    "M11.SI": "",
    "LYY.SI": "",
    "M03.SI": "",
    "8A1.SI": "",
    "BEH.SI": "",
    "AWK.SI": "",
    "580.SI": "",
    "1F1.SI": "",
    "1A0.SI": "",
    "9QX.SI": "",
    "584.SI": "",
    "5QR.SI": "",
    "1H3.SI": "",
    "M15.SI": "",
    "40N.SI": "",
    "585.SI": "",
    "5FX.SI": "",
    "5EW.SI": "",
    "1D3.SI": "",
    "QS9.SI": "",
    "5G4.SI": "",
    "5WV.SI": "",
    "8YY.SI": "",
    "5F4.SI": "",
    "41T.SI": "",
    "V2Y.SI": "",
    "49B.SI": "",
    "5TJ.SI": "",
    "1F0.SI": "",
    "V3M.SI": "",
    "5OC.SI": "",
    "GU5.SI": "",
    "AZA.SI": "",
    "KUX.SI": "",
    "42N.SI": "",
    "5DX.SI": "",
    "J03.SI": "",
    "5HH.SI": "",
    "1H2.SI": "",
    "5CR.SI": "",
    "5IF.SI": "",
    "43E.SI": "",
    "BKV.SI": "",
    "BLU.SI": "",
    "532.SI": "",
    "5QY.SI": "",
    "43F.SI": "",
    "504.SI": "",
    "5UA.SI": "",
    "5VP.SI": "",
    "TVV.SI": "",
    "OMK.SI": "",
    "WJ9.SI": "",
    "508.SI": "",
    "570.SI": "",
    "NHD.SI": "",
    "5OQ.SI": "",
    "43A.SI": "",
    "5SY.SI": "",
    "BLZ.SI": "",
    "5EF.SI": "",
    "BCD.SI": "",
    "5PF.SI": "",
    "BAC.SI": "",
    "5EV.SI": "",
    "5G9.SI": "",
    "5NF.SI": "",
    "1L2.SI": "",
    "S3N.SI": "",
    "583.SI": "",
    "SEJ.SI": "",
    "5EB.SI": "",
    "505.SI": "",
    "BTX.SI": "",
    "ENV.SI": "",
    "SES.SI": "",
    "SJY.SI": "",
    "QZG.SI": "",
    "AWG.SI": "",
    "5LE.SI": "",
    "Y06.SI": "",
    "XCF.SI": "",
    "F10.SI": "",
    "N32.SI": "",
    "BAI.SI": "",
    "V8Y.SI": "",
    "41F.SI": "",
    "5AB.SI": "",
    "BJZ.SI": "",
    "5AI.SI": "",
    "CTO.SI": "",
    "A52.SI": "",
    "5BI.SI": "",
    "E6R.SI": "",
    "BDU.SI": "",
    "E27.SI": "",
    "5RA.SI": "",
    "AOF.SI": "",
    "AAJ.SI": "",
    "5KI.SI": "",
    "594.SI": "",
    "R14.SI": "",
    "5AU.SI": "",
    "5EG.SI": "",
    "5AL.SI": "",
    "AWC.SI": "",
    "BJV.SI": "",
    "BQC.SI": "",
    "CIN.SI": "",
    "MF6.SI": "",
    "BNE.SI": "",
    "PRH.SI": "",
    "BAZ.SI": "",
    "FRQ.SI": "",
    "595.SI": "",
    "EHG.SI": "",
    "42F.SI": "",
    "A04.SI": "",
    "RXS.SI": "",
    "BHU.SI": "",
    "NR7.SI": "",
    "42T.SI": "",
    "CHJ.SI": "",
    "G0I.SI": "",
    "1H8.SI": "",
    "1Y1.SI": "",
    "5G2.SI": "",
    "BVQ.SI": "",
    "HQU.SI": "",
    "BTG.SI": "",
    "M05.SI": "",
    "1B0.SI": "",
    "D8DU.SI": "",
    "LS9.SI": "",
    "K29.SI": "",
    "5AE.SI": "",
    "Q0X.SI": "",
    "5HV.SI": "",
    "BLH.SI": "",
    "F86.SI": "",
    "ZB9.SI": "",
    "566.SI": "",
    "42E.SI": "",
    "5SO.SI": "",
    "C9Q.SI": "",
    "S44.SI": "",
    "8K7.SI": "",
    "U77.SI": "",
    "AWI.SI": "",
    "RQ1.SI": "",
    "546.SI": "",
    "C76.SI": "",
    "B69.SI": "",
    "BQD.SI": "",
    "533.SI": "",
    "G50.SI": "",
    "1J5.SI": "",
    "9G2.SI": "",
    "5ML.SI": "",
    "F13.SI": "",
    "S19.SI": "",
    "L02.SI": "",
    "5TP.SI": "",
    "B49.SI": "",
    "40V.SI": "",
    "V7R.SI": "",
    "1F2.SI": "",
    "S23.SI": "",
    "ZKX.SI": "",
    "N08.SI": "",
    "ER0.SI": "",
    "KUO.SI": "",
    "5CF.SI": "",
    "L19.SI": "",
    "564.SI": "",
    "M14.SI": "",
    "BQM.SI": "",
    "WKS.SI": "",
    "1J0.SI": "",
    "L38.SI": "",
    "XJB.SI": "",
    "BCY.SI": "",
    "T12.SI": "",
    "1J4.SI": "",
    "T13.SI": "",
    "1D1.SI": "",
    "BLS.SI": "",
    "BTP.SI": "",
    "5WA.SI": "",
    "I07.SI": "",
    "5WH.SI": "",
    "5VS.SI": "",
    "5IC.SI": "",
    "CLN.SI": "",
    "5DP.SI": "",
    "S7P.SI": "",
    "BPF.SI": "",
    "MR7.SI": "",
    "5WJ.SI": "",
    "41O.SI": "",
    "XZL.SI": "",
    "IX2.SI": "",
    "S7OU.SI": "",
    "OAJ.SI": "",
    "C33.SI": "",
    "BEW.SI": "",
    "A30.SI": "",
    "42R.SI": "",
    "URR.SI": "",
    "D03.SI": "",
    "1MZ.SI": "",
    "QNS.SI": "",
    "5UL.SI": "",
    "A05.SI": "",
    "J2T.SI": "",
    "D5IU.SI": "",
    "U09.SI": "",
    "500.SI": "",
    "DM0.SI": "",
    "5GD.SI": "",
    "MXNU.SI": "",
    "B73.SI": "",
    "JLB.SI": "",
    "BTM.SI": "",
    "5UF.SI": "",
    "DU4.SI": "",
    "TCU.SI": "",
    "579.SI": "",
    "H12.SI": "",
    "5LY.SI": "",
    "BTOU.SI": "",
    "40T.SI": "",
    "LVR.SI": "",
    "5DD.SI": "",
    "Z59.SI": "",
    "F1E.SI": "",
    "S35.SI": "",
    "G20.SI": "",
    "AWZ.SI": "",
    "BN2.SI": "",
    "QC7.SI": "",
    "OXMU.SI": "",
    "5JK.SI": "",
    "BMGU.SI": "",
    "528.SI": "",
    "N02.SI": "",
    "CMOU.SI": "",
    "E3B.SI": "",
    "B58.SI": "",
    "ODBU.SI": "",
    "HLS.SI": "",
    "F83.SI": "",
    "BTE.SI": "",
    "BBW.SI": "",
    "NPW.SI": "",
    "544.SI": "",
    "PPC.SI": "",
    "BDX.SI": "",
    "T24.SI": "",
    "5UX.SI": "",
    "A34.SI": "",
    "Y03.SI": "",
    "P52.SI": "",
    "RE4.SI": "",
    "M01.SI": "",
    "QES.SI": "",
    "B28.SI": "",
    "BHK.SI": "",
    "T6I.SI": "",
    "1D0.SI": "",
    "M1GU.SI": "",
    "S85.SI": "",
    "UD1U.SI": "",
    "MV4.SI": "",
    "AWX.SI": "",
    "U13.SI": "",
    "5JS.SI": "",
    "S56.SI": "",
    "DHLU.SI": "",
    "TSH.SI": "",
    "WJP.SI": "",
    "Q01.SI": "",
    "5IG.SI": "",
    "F9D.SI": "",
    "P34.SI": "",
    "O10.SI": "",
    "C41.SI": "",
    "F03.SI": "",
    "P9D.SI": "",
    "BWM.SI": "",
    "H18.SI": "",
    "MZH.SI": "",
    "5G3.SI": "",
    "E28.SI": "",
    "H07.SI": "",
    "OYY.SI": "",
    "AW9U.SI": "",
    "OU8.SI": "",
    "BEC.SI": "",
    "H22.SI": "",
    "U9E.SI": "",
    "S20.SI": "",
    "CHZ.SI": "",
    "H30.SI": "",
    "NO4.SI": "",
    "UD2.SI": "",
    "G92.SI": "",
    "558.SI": "",
    "S61.SI": "",
    "8AZ.SI": "",
    "Z25.SI": "",
    "DCRU.SI": "",
    "EH5.SI": "",
    "P15.SI": "",
    "CRPU.SI": "",
    "ACV.SI": "",
    "CWBU.SI": "",
    "LJ3.SI": "",
    "5CP.SI": "",
    "B61.SI": "",
    "AGS.SI": "",
    "C70.SI": "",
    "W05.SI": "",
    "S08.SI": "",
    "O5RU.SI": "",
    "S41.SI": "",
    "A26.SI": "",
    "NS8U.SI": "",
    "ADN.SI": "",
    "P40U.SI": "",
    "YF8.SI": "",
    "J85.SI": "",
    "H13.SI": "",
    "P8Z.SI": "",
    "NC2.SI": "",
    "AU8U.SI": "",
    "CWCU.SI": "",
    "A50.SI": "",
    "Q5T.SI": "",
    "AP4.SI": "",
    "U10.SI": "",
    "P7VU.SI": "",
    "JYEU.SI": "",
    "CY6U.SI": "",
    "BSL.SI": "",
    "F17.SI": "",
    "TS0U.SI": "",
    "F99.SI": "",
    "H15.SI": "",
    "M04.SI": "",
    "AIY.SI": "",
    "CC3.SI": "",
    "EB5.SI": "",
    "OV8.SI": "",
    "J91U.SI": "",
    "H02.SI": "",
    "C2PU.SI": "",
    "SK6U.SI": "",
    "D01.SI": "",
    "T14.SI": "",
    "U06.SI": "",
    "T15.SI": "",
    "S59.SI": "",
    "BVA.SI": "",
    "TSCD.SI": "",
    "A7RU.SI": "",
    "C52.SI": "",
    "TQ5.SI": "",
    "E5H.SI": "",
    "CJLU.SI": "",
    "HMN.SI": "",
    "K71U.SI": "",
    "AJBU.SI": "",
    "T82U.SI": "",
    "V03.SI": "",
    "J69U.SI": "",
    "VC2.SI": "",
    "BUOU.SI": "",
    "U14.SI": "",
    "C09.SI": "",
    "S58.SI": "",
    "5E2.SI": "",
    "EMI.SI": "",
    "M44U.SI": "",
    "ME8U.SI": "",
    "N2IU.SI": "",
    "H78.SI": "",
    "U96.SI": "",
    "J36.SI": "",
    "BS6.SI": "",
    "G13.SI": "",
    "C07.SI": "",
    "NIO.SI": "",
    "BN4.SI": "",
    "S68.SI": "",
    "A17U.SI": "",
    "Y92.SI": "",
    "S63.SI": "",
    "TKKD.SI": "",
    "C38U.SI": "",
    "9CI.SI": "",
    "S07.SI": "",
    "Q0F.SI": "",
    "C6L.SI": "",
    "F34.SI": "",
    "K6S.SI": "",
    "TGED.SI": "",
    "TADD.SI": "",
    "TDED.SI": "",
    "U11.SI": "",
    "Z74.SI": "",
    "Z77.SI": "",
    "O39.SI": "",
    "D05.SI": "",
    "UIX.SI": "",
    "43B.SI": "",
    "FQ7.SI": "",
    "554.SI": "",
    "BRD.SI": "",
    "C04.SI": "",
    "541.SI": "",
    "1E3.SI": "",
    "5TT.SI": "",
    "5F7.SI": "",
    "BBP.SI": "",
    "42C.SI": "",
    "AVX.SI": "",
    "41B.SI": "",
    "OTX.SI": "",
    "5GZ.SI": "",
    "BFU.SI": "",
    "B9S.SI": "",
    "BKW.SI": "",
    "5GI.SI": "",
    "K03.SI": "",
    "GRQ.SI": "",
    "1D4.SI": "",
    "1V3.SI": "",
    "5PO.SI": "",
    "OTS.SI": "",
    "A33.SI": "",
    "C06.SI": "",
    "BIX.SI": "",
    "BFI.SI": "",
    "ZXY.SI": "",
    "Y35.SI": "",
    "N01.SI": "",
    "BHD.SI": "",
    "H20.SI": "",
    "5I4.SI": "",
    "CEDU.SI": "",
    "BTJ.SI": "",
    "1A1.SI": "",
    "LMS.SI": "",
    "42W.SI": "",
    "S71.SI": "",
    "1D5.SI": "",
    "AYN.SI": "",
    "BXE.SI": "",
    "CNE.SI": "",
    "5PC.SI": "",
    "Y8E.SI": "",
    "AJ2.SI": "",
    "T41.SI": "",
    "596.SI": "",
    "543.SI": "",
    "BIP.SI": "",
    "540.SI": "",
    "O08.SI": "",
    "P36.SI": "",
    "L23.SI": "",
    "C05.SI": "",
    "P8A.SI": "",
    "B26.SI": "",
    "C13.SI": "",
    "I06.SI": "",
    "BDA.SI": "",
    "5NV.SI": "",
    "A31.SI": "",
    "SGR.SI": "",
    "Y3D.SI": "",
    "TWL.SI": "",
    "KJ5.SI": "",
    "5WF.SI": "",
    "NEX.SI": "",
    "VI2.SI": "",
    "1B1.SI": "",
    "YK9.SI": "",
    "I49.SI": "",
    "BQF.SI": "",
    "T55.SI": "",
    "5G1.SI": "",
    "1R6.SI": "",
    "NLC.SI": "",
    "42L.SI": "",
    "C8R.SI": "",
    "9I7.SI": "",
    "A55.SI": "",
    "S69.SI": "",
    "5MZ.SI": "",
    "O9E.SI": "",
    "5OI.SI": "",
    "BDR.SI": "",
    "1F3.SI": "",
    "BEZ.SI": "",
    "1AZ.SI": "",
    "569.SI": "",
    "S29.SI": "",
    "NXR.SI": "",
    "BKX.SI": "",
    "5DM.SI": "",
    "WPC.SI": "",
    "K75.SI": "",
    "BKA.SI": "",
    # "C6L.SI": "Singapore Airlines - Airline",
    # "U96.SI": "Sembcorp Industries - Utilities",
    # "D05.SI": "DBS Group Holdings - Banking",
    # "Z74.SI": "Singtel - Telecommunications",
    # "C07.SI": "Jardine Cycle & Carriage - Automotive",
    # "BN4.SI": "Keppel Corporation - Conglomerate",
    # "F34.SI": "Wilmar International - Agriculture",
    # "BS6.SI": "Yangzijiang Shipbuilding - Shipbuilding",
    # "O39.SI": "OCBC Bank - Banking",
    # "9CI.SI": "CapitaLand - Real Estate",
    # "U11.SI": "United Overseas Bank - Banking",
    # "N2IU.SI": "Mapletree Commercial Trust - Real Estate",
    # "H78.SI": "Hongkong Land Holdings - Real Estate",
    # "S68.SI": "Singapore Exchange - Financial Services",
    # "A17U.SI": "Ascendas REIT - Real Estate",
    # "C09.SI": "City Developments - Real Estate",
    # "M44U.SI": "Mapletree Logistics Trust - Real Estate",
    # "V03.SI": "Venture Corporation - Electronics",
    # "J69U.SI": "Frasers Logistics & Commercial Trust - Real Estate",
    # "ME8U.SI": "Mapletree Industrial Trust - Real Estate",
    # "AJBU.SI": "Ascott Residence Trust - Hospitality",
    # "NOBGY": "Noble Group - Commodities",
    # "K71U.SI": "Keppel REIT - Real Estate",
    # "G13.SI": "Genting Singapore - Hospitality",
    # "Y92.SI": "Thai Beverage - Beverages",
    # "Q5T.SI": "Far East Hospitallity Trust - REITS",
    # "U14.SI": "UOL Group - Real Estate",
    # "S63.SI": "ST Engineering - Engineering",
    # "NS8U.SI": "Hutchison Port Holdings Trust  - REITS",
    # "C52.SI": "ComfortDelGro - Transportation",
    # "H02.SI": "Haw Par Corporation Limited ",
    # "E5H.SI": "Golden Agri-Resources - Agriculture",
    # "S59.SI": "SIA Engineering - Aerospace",
    # "U09.SI": "Avarga Limited",
    # "O5RU.SI": "AIMS APAC REIT - Real Estate",
    # "BUOU.SI": "Frasers Logistics & Commercial Trust  - Real Estate",
    # "M1GU.SI": "Sabana Industrial Real Estate Investment Trust - Real Estate",
    # "BVA.SI": "Top Glove Corporation Bhd.",
    # "D03.SI": "Del Monte Pacific - Food & Beverage",
    # "P40U.SI": "Starhill Global Real Estate Investment Trust ",
    # "F9D.SI": "Boustead Singapore Limited",
    # "T82U.SI": "Suntec REIT - Real Estate",
    # "TQ5.SI": "Frasers Property - Real Estate",
    # "BSL.SI": "Raffles Medical Group - Healthcare",
    # "1D4.SI": "Aoxin Q & M Dental Group Limited",
    # "NR7.SI": "Raffles Education Corporation - Education",
    # "5CP.SI": "Silverlake Axis - Technology",
    # "T14.SI": "Tianjin Pharmaceutical Da Ren Tang Group Corporation Limited",
    # "C2PU.SI": "Parkway Life Real Estate Investment Trust ",
    # "U96.SI": "Sembcorp Industries - Energy",
    # "5TP.SI": "CNMC Goldmine Holdings Limited",
    # "G07.SI": "Great Eastern - Insurance",
    # "TQ5.SI": "Frasers Property Limited ",
    # "CC3.SI": "StarHub - Telecommunications",
    # "BN4.SI": "Keppel Corporation - Industrial Conglomerate",
    # "C09.SI": "City Developments Limited - Real Estate",
    # "O10.SI": "Far East Orchard - Real Estate",
    # "S58.SI": "SATS Ltd. - Aviation",
    # "Y06.SI": "Green Build Technology Limited",
    # "Z25.SI": "Yanlord Land Group - Real Estate",
    # "Z59.SI": "Yoma Strategic Holdings - Conglomerate",
    # "C76.SI": "Creative Technology - Technology",
    # "A50.SI": "Thomson Medical Group - Healthcare",
    # "5IG.SI": "Gallant Venture Ltd",
    # "1C0.SI": "Emerging Towns & Cities Singapore Ltd.",
    # "C38U.SI": "CapitaLand Integrated Commercial Trust - Real Stack",
    # "C52.SI": "ComfortDelGro Corporation - Transportation",
    # "G92.SI": "China Aviation Oil (Singapore) Corporation Ltd",
    # "CRPU.SI": "Sasseur REIT - Real Estate",
    # "H30.SI": "Hong Fok Corporation Limited",
    # "H13.SI": "Ho Bee Land - Real Estate",
    # "1B1.SI": "HC Surgical Specialists - Healthcare",
    # "AWX.SI": "AEM Holdings Ltd.",
    # "1J5.SI": "Hyphens Pharma - Pharmaceuticals",
    # "B61.SI": "Bukit Sembawang Estates - Real Estate",
    # "A7RU.SI": "Keppel Infrastructure Trust",
    # "BTOU.SI": "Manulife US REIT - Real Estate",
    # "BDA.SI": "PNE Industries Ltd",
    # "9I7.SI": "No Signboard Holdings - Food & Beverage",
    # "BWCU.SI": "EC World Real Estate Investment Trust ",
    # "S7OU.SI": "Asian Pay Television Trust",
    # "TS0U.SI": "OUE Commercial REIT - Real Estate",
    # "U9E.SI": "China Everbright Water Limited",
    # "S8N.SG": "Sembcorp Marine - Marine",
    # "5GZ.SI": "HGH Holdings Ltd.",
    # "RE4.SI": "Geo Energy Resources - Energy",
    # "40T.SI": "ISEC Healthcare Ltd.",
    # "U77.SI": "Sarine Technologies - Technology",
    # "AJ2.SI": "Ouhua Energy Holdings Limited",
    # "1A4.SI": "AGV Group - Industrial",
    # "S41.SI": "Hong Leong Finance Limited",
    # "Q0X.SI": "Ley Choon Group - Construction",
    # "S71.SI": "Sunright Limited",
    # "5UX.SI": "Oxley Holdings - Real Estate",
    # "5IF.SI": "Natural Cool Holdings Limited - Hospitality",
    # "OV8.SI": "Sheng Siong Group - Retail",
    # "AIY.SI": "iFast Corporation - Financial Services",
    # "5CP.SI": "Silverlake Axis - Technology",
    # "P15.SI": "Pacific Century Regional Developments Limited",
    # "5AB.SI": "Trek 2000 International Ltd",
    # "AZI.SI": "AusNet Services - Utilities",
    # "U13.SI": "United Overseas Insurance Limited",
    # "558.SI": "UMS Holdings - Semiconductors",
    # "1D0.SI": "Kimly Limited",
    # "I07.SI": "ISDN Holdings - Industrial Automation",
    # "5UX.SI": "Oxley Holdings Limited",
    # "M35.SI": "Wheelock Properties - Real Estate",
    # "A30.SI": "Aspial Corporation Limited",
    # "5G1.SI": "EuroSports Global Limited",
    # "BJZ.SI": "Koda Ltd - Manufacturing",
    # "5TT.SI": "Keong Hong Holdings Limited",
}

stock_malaysia = {
    # UEM
    "5148.KL": "UEM Sunrise Berhad",
    "8583.KL": "Mah Sing Group Berhad",  # check
    "8567.KL": "Eco World Development Group Berhad",  # check
    "5299.KL": "S P Setia Berhad",
    "3743.KL": "IOI Properties Group Berhad",  # check
    "3336.KL": "Sunway Berhad",
    "5202.KL": "I-Berhad",
    "5053.KL": "OSK Holdings Berhad",
    "7164.KL": "LBS Bina Group Berhad",
    "5205.KL": "Matrix Concepts Holdings Berhad",
    "1724.KL": "Paramount Corporation Berhad",
    "1171.KL": "Crescendo Corporation Berhad",
    "7123.KL": "Tambun Indah Land Berhad",
    "5184.KL": "WCT Holdings Berhad",
    "3417.KL": "Eastern & Oriental Berhad",
    "5182.KL": "Avaland Berhad",
    "0166.KL": "Inari Amertron Berhad",
    "5008.KL": "Tropicana Corporation Berhad",
    "5200.KL": "UOA Development Bhd",
    "5038.KL": "KSL Holdings Berhad",
    "5239.KL": "Titijaya Land Berhad",
    "5075.KL": "Plenitude Berhad",
    "5175.KL": "Ivory Properties Group Berhad",
    "5020.KL": "Glomac Berhad",
    "1651.KL": "Malaysian Resources Corporation Berhad",
    "5062.KL": "Hua Yang Berhad",
    "6076.KL": "Encorp Berhad",
    # YTL
    "4677.KL": "YTL Corporation Berhad",
    "6742.KL": "YTL Power International Berhad",
    "5109.KL": "YTL Hospitality REIT",
    "P40U.SI": "Starhill Global REIT",
    # ZEN List
    "GRAB": "GRAB holdings",
    "5318.KL": "DXN holdings",
    "1023.KL": "CIMB",
    # OTher's maybe
    "5183.KL": "Petronas Chemicals Group",
    "5347.KL": "Tenaga Nasional Berhad",
    "7293.KL": "Yinson Holdings Berhad",
    "7033.KL": "Dialog Group Berhad",
    "5196.KL": "Bumi Armada Berhad",
    "5279.KL": "Serba Dinamik Holdings Berhad",
    "5204.KL": "Petronas Gas Berhad",
    "5209.KL": "Gas Malaysia Berhad",
    "2445.KL": "IOI Corporation Berhad",
    "5185.KL": "Sime Darby Plantation Berhad",
    "3476.KL": "MMC Corporation Berhad",
    "2828.KL": "IJM Corporation Berhad",
    "0018.KL": "GHL Systems Berhad",
    "0023.KL": "Datasonic Group Berhad",
    "0138.KL": "MyEG Services Berhad",
    "0041.KL": "Revenue Group Berhad",
    "0082.KL": "Green Packet Berhad",
    "0216.KL": "Vitrox Corporation Berhad",
    "0072.KL": "Inari Amertron Berhad",
    "1155.KL": "Malayan Banking Berhad (Maybank)",
    "1295.KL": "Public Bank Berhad",
    "6888.KL": "RHB Bank Berhad",
    "5819.KL": "Hong Leong Bank Berhad",
    "0146.KL": "Dataprep Holdings Berhad",
    "7183.KL": "IHH Healthcare Berhad",
    "5285.KL": "Frontken Corporation Berhad",
    "3867.KL": "Axiata Group Berhad",
    "6012.KL": "Maxis Berhad",
    # 工业
    "7113.KL": "Top Glove Corporation Berhad",  # 工业产品
    "5168.KL": "Hartalega Holdings Berhad",  # 工业产品
    "7153.KL": "Kossan Rubber Industries Berhad",  # 工业产品
    "7106.KL": "Supermax Corporation Berhad",  # 工业产品
    # 材料
    "8869.KL": "Press Metal Aluminium Holdings Berhad",  # 金属
    "6556.KL": "Ann Joo Resources Berhad",  # 钢铁
    "3794.KL": "Malayan Cement Berhad",  # 建材
    "4065.KL": "PPB Group Berhad",  # 多元化材料
    # 必需消费品
    "4707.KL": "Nestle (Malaysia) Berhad",  # 食品
    "3689.KL": "Fraser & Neave Holdings Bhd",  # 饮料
    "7084.KL": "QL Resources Berhad",  # 农产品
    "3026.KL": "Dutch Lady Milk Industries Berhad",  # 乳制品
    "7216.KL": "Kawan Food Berhad",  # 食品
    # 非必需消费品
    "4715.KL": "Genting Malaysia Berhad",  # 娱乐
    "1562.KL": "Berjaya Sports Toto Berhad",  # 娱乐
    "6947.KL": "Digi.Com Berhad",  # 通信
    "1066.KL": "RHB Bank Berhad",  # 银行
    # 公用事业
    "5264.KL": "MALAKOFF CORPORATION BERHAD",  # 电力
    # 运输
    "5246.KL": "Westports Holdings Berhad",  # 港口运营
    "5014.KL": "Malaysia Airports Holdings Berhad",  # 机场运营
    "3816.KL": "MISC Berhad",  # 航运
    "5099.KL": "AirAsia Group Berhad",  # 航空
    "0078.KL": "GDEX Berhad",  # 物流
}


def fast_scanning():
    market_client = MarketClient(init_log=True)
    list_obj = market_client.get_market_tickers()
    symbols = []
    for obj in list_obj:
        amount = obj.amount
        count = obj.count
        open = obj.open
        close = obj.close
        low = obj.low
        high = obj.high
        vol = obj.vol
        symbol = obj.symbol
        bid = obj.bid
        bidSize = obj.bidSize
        ask = obj.ask
        askSize = obj.askSize
        # if (amount * close >= 10000000):
        if vol >= 5000000:
            symbols.append(symbol)
    return symbols


def count_obv_cross(IMarketSensorImp, ccy, interval, sample, show=False):
    params.update(
        {
            "interval": interval,
            "funds": 100,
            "stake_cap": 50,
            "symbol": Symbol(ccy),
        }
    )
    sensor = IMarketSensorImp(symbol=params.get("symbol"), interval=interval)
    df = sensor.scan(sample)
    if len(df) == 0:
        print(f"{ccy} no data")
        return 0
    df = sensor.fetch(df)

    df.to_csv(f"{DATA_DIR}/{ccy}_cached.csv", index=True)
    print(f"{DATA_DIR}/{ccy}_cached.csv")
    sp = StrategyParam(**params)
    scout = TurtleScout(params=sp)
    df = scout._calc_OBV(df, multiplier=3)
    if df.iloc[-1].OBV_UP:
        print("============= OBV UP ============")
    print(f"{ccy=}: {len(df[df.OBV_UP])}")
    if show and df[-90:].OBV_UP.any():
        chart(df[-90:], ccy)
    return len(df[df.OBV_UP])


def chart(df, code):
    # Ensure the index is of type DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        df["Date"] = pd.to_datetime(df["Date"])
        df.set_index("Date", inplace=True)
        df.index = pd.to_datetime(df.index)

    # Create columns for scatter plots with NaN where there's no marker
    obv_up_marker = np.where(df["OBV_UP"], df["OBV"], np.nan)
    close_up_marker = np.where(df["OBV_UP"], df["Close"], np.nan)

    # Prepare additional plots
    add_plots = [
        mpf.make_addplot(
            df["OBV"], panel=1, color="g", secondary_y=False, ylabel="OBV"
        ),
        mpf.make_addplot(df["upper_bound"], panel=1, color="gray", linestyle="--"),
        mpf.make_addplot(df["lower_bound"], panel=1, color="gray", linestyle="--"),
        mpf.make_addplot(
            obv_up_marker,
            panel=1,
            type="scatter",
            markersize=100,
            marker="^",
            color="red",
        ),
        mpf.make_addplot(
            close_up_marker, type="scatter", markersize=100, marker="^", color="y"
        ),
    ]

    # Create a candlestick chart with additional plots
    mpf.plot(
        df,
        type="candle",
        addplot=add_plots,
        title=f"{stock_malaysia.get(code, code)} Price and OBV with Bounds",
        ylabel="Price (USD)",
        style="yahoo",
        datetime_format="%Y-%m-%d %H:%M:%S",
    )


@click.command()
@click.option(
    "--market",
    type=click.Choice(["my", "sg", "crypto"]),
    required=True,
    help="选择市场类型",
)
@click.option("--symbol", default=None, help="指定交易对或股票代码")
@click.option(
    "--interval",
    default="1day",
    help="交易间隔: 1min, 5min, 15min, 30min, 60min, 4hour, 1day, 1mon, 1week, 1year",
)
@click.option("--show", is_flag=False, help="是否显示图表")
@click.option("--backtest", is_flag=False, help="是否进行回测")
def main(market, symbol, interval, show, backtest):
    if market in ("my", "sg"):
        sensor_cls = YahooMarketSensor
        if symbol:
            symbols = [symbol]
        else:
            symbols = (
                stock_malaysia.keys() if market == "my" else stocks_singapore.keys()
            )
        sample = 365 * 20  # 20 years
    elif market == "crypto":
        sensor_cls = HuobiMarketSensor
        symbols = [symbol] if symbol else fast_scanning()
        sample = 240 * 3  # 3 years

    symbols_count = {
        s: count_obv_cross(sensor_cls, s, interval, sample, show) for s in symbols
    }

    if backtest:
        # 进行回测逻辑
        perform_backtest(symbols_count)

    c = Counter(symbols_count)
    from pprint import pprint

    pprint(c.most_common())


def perform_backtest(symbols_count):
    result = {}
    for code in symbols_count.keys():
        try:
            result[code] = entry(code, "1day", 100, 10.5)
        except Exception as e:
            print(f"Back test error: {code}, {e}")
    performance_review(result)


def performance_review(backtest_results):
    # 将数据转换为 DataFrame
    df = pd.DataFrame(list(backtest_results.items()), columns=["Stock", "Return"])
    df.to_csv(f"{REPORTS_DIR}/backtest.csv")
    df = df[df.Return != 0]
    print(df[:60])

    # 总体表现
    average_return = df["Return"].mean()
    total_return = df["Return"].sum()

    # 风险分析
    std_dev = df["Return"].std()
    max_drawdown = df["Return"].min()

    # 胜率分析
    positive_returns = df[df["Return"] > 0].shape[0]
    negative_returns = df[df["Return"] < 0].shape[0]
    win_rate = positive_returns / df.shape[0]
    loss_rate = negative_returns / df.shape[0]

    # Sharpe Ratio
    risk_free_rate = 0.0
    sharpe_ratio = (df["Return"].mean() - risk_free_rate) / df["Return"].std()

    # Sortino Ratio
    negative_returns = df[df["Return"] < 0]["Return"]
    downside_std = negative_returns.std()
    sortino_ratio = (df["Return"].mean() - risk_free_rate) / downside_std

    # 打印结果
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Sortino Ratio: {sortino_ratio:.2f}")

    # 打印结果
    print(f"平均回报率: {average_return:.2f}%")
    print(f"总回报率: {total_return:.2f}%")
    print(f"回报率标准差: {std_dev:.2f}%")
    print(f"最大回撤: {max_drawdown:.2f}%")
    print(f"胜率: {win_rate:.2f}")
    print(f"失败率: {loss_rate:.2f}")

    # 绘制收益分布图
    plt.figure(figsize=(14, 8))
    sns.histplot(df["Return"], bins=20, kde=True, color="blue")
    plt.title("Distribution of Returns")
    plt.xlabel("Return (%)")
    plt.ylabel("Frequency")
    plt.show()


from .sensor.market_sensor import MongoDBHandler


def x():
    params.update(
        {
            "funds": 100,
            "stake_cap": 50,
            "symbol": Symbol("btcusdt"),
        }
    )
    sensor = HuobiMarketSensor(symbol=params["symbol"], interval="1min")
    df = sensor.scan(2000)
    mongo = MongoDBHandler(collection_name=f"{params['symbol'].name}_raw")
    mongo.save(df)
    print(df)
    # df = sensor.fetch(df)
    # df.to_csv(f"{DATA_DIR}/{ccy}_cached.csv", index=True)


def y():
    params.update(
        {
            "funds": 100,
            "stake_cap": 50,
            "symbol": Symbol("btcusdt"),
        }
    )
    mongo = MongoDBHandler(collection_name=f"{params['symbol'].name}_raw")
    df = mongo.load()
    print(df)
    # df = sensor.fetch(df)
    # df.to_csv(f"{DATA_DIR}/{ccy}_cached.csv", index=True)


if __name__ == "__main__":
    # result = {code: entry(code, "1day", 100, 50) for code in stock_malaysia.keys()}
    # from pprint import pprint

    # pprint(result)
    main()
    # y()
