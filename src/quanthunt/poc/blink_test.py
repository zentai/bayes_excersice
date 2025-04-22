from concurrent.futures import ProcessPoolExecutor, as_completed
from . import cloud_story
from .hunterverse.interface import Symbol, StrategyParam
from config import config
import pandas as pd
import multiprocessing

DATA_DIR, SRC_DIR, REPORTS_DIR = config.data_dir, config.src_dir, config.reports_dir


# Function to handle processing for each stock symbol (this runs in a separate process)
def process_stock(stock_symbol):
    try:
        cloud_story.params.update(
            {
                "interval": "1day",
                "funds": 100,
                "stake_cap": 100,
                "symbol": Symbol(stock_symbol),
            }
        )
        sp = StrategyParam(**cloud_story.params)
        df, review = cloud_story.start_journey(sp)
        print(sp)
        print(review)

        # Convert review DataFrame to dictionary (assuming it's a single row)
        review_dict = review.to_dict(orient="records")[
            0
        ]  # Convert DataFrame to a dictionary

        # Add symbol and path to the review dictionary
        review_dict["Symbol"] = sp.symbol.name
        review_dict["path"] = f"{REPORTS_DIR}/{sp}.csv"

        return review_dict  # Return the review dictionary
    except Exception as e:
        print(f"Error processing {stock_symbol}: {e}")
        return None


if __name__ == "__main__":
    # Stock list
    # stock_list = "0018.KL,1023.KL,3026.KL,4677.KL,5099.KL,5200.KL,5299.KL,7033.KL,8567.KL,0023.KL,1066.KL,3336.KL,4707.KL,5109.KL,5202.KL,5318.KL,7084.KL,8583.KL,0041.KL,1155.KL,3417.KL,4715.KL,5148.KL,5204.KL,5347.KL,7106.KL,8869.KL,0072.KL,1171.KL,3476.KL,5008.KL,5168.KL,5205.KL,5819.KL,7113.KL,0078.KL,1295.KL,3689.KL,5014.KL,5175.KL,5209.KL,6012.KL,7123.KL,0082.KL,1562.KL,3743.KL,5020.KL,5182.KL,5239.KL,6076.KL,7153.KL,0138.KL,1651.KL,3794.KL,5038.KL,5183.KL,5246.KL,6556.KL,7164.KL,0146.KL,1724.KL,3816.KL,5053.KL,5184.KL,5264.KL,6742.KL,7183.KL,0166.KL,2445.KL,3867.KL,5062.KL,5185.KL,5279.KL,6888.KL,7216.KL,0216.KL,2828.KL,4065.KL,5075.KL,5196.KL,5285.KL,6947.KL,7293.KL".split(
    # ","
    # )
    # stock_list = "AIY.SI,M04.SI,CC3.SI,J91U.SI,EB5.SI,OV8.SI,H02.SI,D01.SI,SK6U.SI,C2PU.SI,BVA.SI,U06.SI,T14.SI,T15.SI,S59.SI,A7RU.SI,TSCD.SI,C52.SI,TQ5.SI,E5H.SI,HMN.SI,CJLU.SI,K71U.SI,T82U.SI,AJBU.SI,V03.SI,J69U.SI,BUOU.SI,VC2.SI,U14.SI,C09.SI,S58.SI,5E2.SI,EMI.SI,M44U.SI,ME8U.SI,N2IU.SI,H78.SI,U96.SI,J36.SI,BS6.SI,G13.SI,C07.SI,BN4.SI,NIO.SI,S68.SI,A17U.SI,Y92.SI,S63.SI,TKKD.SI,9CI.SI,C38U.SI,S07.SI,Q0F.SI,C6L.SI,F34.SI,K6S.SI,TGED.SI,TADD.SI,TDED.SI,U11.SI,Z77.SI,Z74.SI,O39.SI,D05.SI".split(
    #     ","
    # )
    # stock_list = "VNM.SI GRU.SI LCS.SI SQU.SI SHD.SI YLU.SI TATD.SI S45U.SI SSS.SI CXS.SI ICU.SI YLD.SI ICM.SI ESG.SI JJJ.SI VND.SI MCN.SI TCPD.SI LSS.SI GRO.SI LCU.SI SCY.SI TPED.SI SQQ.SI YWTR.SI W4VR.SI AYV.SI Z4D.SI BJD.SI YYB.SI KUH.SI BKK.SI BFK.SI UUK.SI 5OR.SI M11.SI LYY.SI M03.SI 8A1.SI BEH.SI AWK.SI 580.SI 1F1.SI 1A0.SI 9QX.SI 584.SI 5QR.SI 1H3.SI M15.SI 40N.SI 585.SI 5FX.SI 5EW.SI 1D3.SI QS9.SI 5G4.SI 5WV.SI 8YY.SI 5F4.SI 41T.SI V2Y.SI 49B.SI 5TJ.SI 1F0.SI V3M.SI 5OC.SI GU5.SI AZA.SI KUX.SI 42N.SI 5DX.SI J03.SI 5HH.SI 1H2.SI 5CR.SI 5IF.SI 43E.SI BKV.SI BLU.SI 532.SI 5QY.SI 43F.SI 504.SI 5UA.SI 5VP.SI TVV.SI OMK.SI WJ9.SI 508.SI 570.SI NHD.SI 5OQ.SI 43A.SI 5SY.SI BLZ.SI 5EF.SI BCD.SI 5PF.SI BAC.SI 5EV.SI 5G9.SI 5NF.SI 1L2.SI S3N.SI 583.SI SEJ.SI 5EB.SI 505.SI BTX.SI ENV.SI SES.SI SJY.SI QZG.SI AWG.SI 5LE.SI Y06.SI XCF.SI F10.SI N32.SI BAI.SI V8Y.SI 41F.SI 5AB.SI BJZ.SI 5AI.SI CTO.SI A52.SI 5BI.SI E6R.SI BDU.SI E27.SI 5RA.SI AOF.SI AAJ.SI 5KI.SI 594.SI R14.SI 5AU.SI 5EG.SI 5AL.SI AWC.SI BJV.SI BQC.SI CIN.SI MF6.SI BNE.SI PRH.SI BAZ.SI FRQ.SI 595.SI EHG.SI 42F.SI A04.SI RXS.SI BHU.SI NR7.SI 42T.SI CHJ.SI G0I.SI 1H8.SI 1Y1.SI 5G2.SI BVQ.SI HQU.SI BTG.SI M05.SI 1B0.SI D8DU.SI LS9.SI K29.SI 5AE.SI Q0X.SI 5HV.SI BLH.SI F86.SI ZB9.SI 566.SI 42E.SI 5SO.SI C9Q.SI S44.SI 8K7.SI U77.SI AWI.SI RQ1.SI 546.SI C76.SI B69.SI BQD.SI 533.SI G50.SI 1J5.SI 9G2.SI 5ML.SI F13.SI S19.SI L02.SI 5TP.SI B49.SI 40V.SI V7R.SI 1F2.SI S23.SI ZKX.SI N08.SI ER0.SI KUO.SI 5CF.SI L19.SI 564.SI M14.SI BQM.SI WKS.SI 1J0.SI L38.SI XJB.SI BCY.SI T12.SI 1J4.SI T13.SI 1D1.SI BLS.SI BTP.SI 5WA.SI I07.SI 5WH.SI 5VS.SI 5IC.SI CLN.SI 5DP.SI S7P.SI BPF.SI MR7.SI 5WJ.SI 41O.SI XZL.SI IX2.SI S7OU.SI OAJ.SI C33.SI BEW.SI A30.SI 42R.SI URR.SI D03.SI 1MZ.SI QNS.SI 5UL.SI A05.SI J2T.SI D5IU.SI U09.SI 500.SI DM0.SI 5GD.SI MXNU.SI B73.SI JLB.SI BTM.SI 5UF.SI DU4.SI TCU.SI 579.SI H12.SI 5LY.SI BTOU.SI 40T.SI LVR.SI 5DD.SI Z59.SI F1E.SI S35.SI G20.SI AWZ.SI BN2.SI QC7.SI OXMU.SI 5JK.SI BMGU.SI 528.SI N02.SI CMOU.SI E3B.SI B58.SI ODBU.SI HLS.SI F83.SI BTE.SI BBW.SI NPW.SI 544.SI PPC.SI BDX.SI T24.SI 5UX.SI A34.SI Y03.SI P52.SI RE4.SI M01.SI QES.SI B28.SI BHK.SI T6I.SI 1D0.SI M1GU.SI S85.SI UD1U.SI MV4.SI AWX.SI U13.SI 5JS.SI S56.SI DHLU.SI TSH.SI WJP.SI Q01.SI 5IG.SI F9D.SI P34.SI O10.SI C41.SI F03.SI P9D.SI BWM.SI H18.SI MZH.SI 5G3.SI E28.SI H07.SI OYY.SI AW9U.SI OU8.SI BEC.SI H22.SI U9E.SI S20.SI CHZ.SI H30.SI NO4.SI UD2.SI G92.SI 558.SI S61.SI 8AZ.SI Z25.SI DCRU.SI EH5.SI P15.SI CRPU.SI ACV.SI CWBU.SI LJ3.SI 5CP.SI B61.SI AGS.SI C70.SI W05.SI S08.SI O5RU.SI S41.SI A26.SI NS8U.SI ADN.SI P40U.SI YF8.SI J85.SI H13.SI P8Z.SI NC2.SI AU8U.SI CWCU.SI A50.SI Q5T.SI AP4.SI U10.SI P7VU.SI JYEU.SI CY6U.SI BSL.SI F17.SI TS0U.SI F99.SI H15.SI M04.SI AIY.SI CC3.SI EB5.SI OV8.SI J91U.SI H02.SI C2PU.SI SK6U.SI D01.SI T14.SI U06.SI T15.SI S59.SI BVA.SI TSCD.SI A7RU.SI C52.SI TQ5.SI E5H.SI CJLU.SI HMN.SI K71U.SI AJBU.SI T82U.SI V03.SI J69U.SI VC2.SI BUOU.SI U14.SI C09.SI S58.SI 5E2.SI EMI.SI M44U.SI ME8U.SI N2IU.SI H78.SI U96.SI J36.SI BS6.SI G13.SI C07.SI NIO.SI BN4.SI S68.SI A17U.SI Y92.SI S63.SI TKKD.SI C38U.SI 9CI.SI S07.SI Q0F.SI C6L.SI F34.SI K6S.SI TGED.SI TADD.SI TDED.SI U11.SI Z74.SI Z77.SI O39.SI D05.SI UIX.SI 43B.SI FQ7.SI 554.SI BRD.SI C04.SI 541.SI 1E3.SI 5TT.SI 5F7.SI BBP.SI 42C.SI AVX.SI 41B.SI OTX.SI 5GZ.SI BFU.SI B9S.SI BKW.SI 5GI.SI K03.SI GRQ.SI 1D4.SI 1V3.SI 5PO.SI OTS.SI A33.SI C06.SI BIX.SI BFI.SI ZXY.SI Y35.SI N01.SI BHD.SI H20.SI 5I4.SI CEDU.SI BTJ.SI 1A1.SI LMS.SI 42W.SI S71.SI 1D5.SI AYN.SI BXE.SI CNE.SI 5PC.SI Y8E.SI AJ2.SI T41.SI 596.SI 543.SI BIP.SI 540.SI O08.SI P36.SI L23.SI C05.SI P8A.SI B26.SI C13.SI I06.SI BDA.SI 5NV.SI A31.SI SGR.SI Y3D.SI TWL.SI KJ5.SI 5WF.SI NEX.SI VI2.SI 1B1.SI YK9.SI I49.SI BQF.SI T55.SI 5G1.SI 1R6.SI NLC.SI 42L.SI C8R.SI 9I7.SI A55.SI S69.SI 5MZ.SI O9E.SI 5OI.SI BDR.SI 1F3.SI BEZ.SI 1AZ.SI 569.SI S29.SI NXR.SI BKX.SI 5DM.SI WPC.SI K75.SI BKA.SI".split(
    #     " "
    # )
    from .market_scanning import stocks_malaysia, stocks_singapore

    stock_list = stocks_singapore.keys()
    # Initialize an empty list to collect reviews
    reviews = []

    # Set up ProcessPoolExecutor with a fixed number of workers (based on CPU cores)
    max_workers = int(
        multiprocessing.cpu_count()
    )  # Use the number of CPU cores as the max workers
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all stock processing tasks to the process pool
        futures = {executor.submit(process_stock, stock): stock for stock in stock_list}

        # Collect the results as they are completed
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                reviews.append(result)  # Add the review (a dictionary) to the list

    # Create a DataFrame from the collected reviews
    if reviews:
        df_reviews = pd.DataFrame(reviews)

        # Ensure the 'Symbol' column is the first column and 'path' is the last column
        cols = (
            ["Symbol"]
            + [col for col in df_reviews.columns if col != "Symbol" and col != "path"]
            + ["path"]
        )
        df_reviews = df_reviews[cols]

        # Output the DataFrame for review
        print(df_reviews)
        cloud_story.params.update(
            {
                "interval": "1day",
                "funds": 1000,
                "stake_cap": 100,
                "symbol": Symbol(list(stock_list)[0]),
            }
        )
        sp = StrategyParam(**cloud_story.params)
        df_reviews.to_csv(
            f"{REPORTS_DIR}/reviews_{str(sp).split('_')[-1]}.csv", index=False
        )
        print(f"created: {REPORTS_DIR}/reviews_{str(sp).split('_')[-1]}.csv")
