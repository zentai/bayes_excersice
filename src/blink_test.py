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
    stock_list = "1A4.SI,5GZ.SI,A7RU.SI,BSL.SI,C6L.SI,G13.SI,M35.SI,P40U.SI,S7OU.SI,U96.SI,1B1.SI,5IF.SI,AIY.SI,BTOU.SI,C76.SI,G92.SI,M44U.SI,P9D.SI,S7P.SI,U9E.SI,1C0.SI,5IG.SI,AJ2.SI,BUOU.SI,CC3.SI,H02.SI,ME8U.SI,Q0X.SI,T14.SI,V03.SI,1D0.SI,5TP.SI,AJBU.SI,BVA.SI,CRPU.SI,H13.SI,N2IU.SI,Q5T.SI,T82U.SI,Y06.SI,1D4.SI,5TT.SI,AWX.SI,BWCU.SI,D03.SI,H20.SI,NR7.SI,RE4.SI,TQ5.SI,Y92.SI,1J5.SI,5UX.SI,AZI.SI,C06.SI,D05.SI,H30.SI,NS8U.SI,S41.SI,TS0U.SI,Z25.SI,40T.SI,9CI.SI,B61.SI,C07.SI,E5H.SI,H78.SI,O10.SI,S58.SI,U09.SI,Z59.SI,558.SI,9I7.SI,BDA.SI,C09.SI,F34.SI,I07.SI,O39.SI,S59.SI,U11.SI,Z74.SI,5AB.SI,A17U.SI,BJZ.SI,C2PU.SI,F9D.SI,J69U.SI,O5RU.SI,S63.SI,U13.SI,5CP.SI,A30.SI,BN4.SI,C38U.SI,FQ7.SI,K71U.SI,OV8.SI,S68.SI,U14.SI,5G1.SI,A50.SI,BS6.SI,C52.SI,G07.SI,M1GU.SI,P15.SI,S71.SI,U77.SI".split(
        ","
    )
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
                "funds": 100,
                "stake_cap": 100,
                "symbol": Symbol(stock_list[0]),
            }
        )
        sp = StrategyParam(**cloud_story.params)
        df_reviews.to_csv(
            f"{REPORTS_DIR}/reviews_{str(sp).split('_')[-1]}.csv", index=False
        )
        print(f"created: {REPORTS_DIR}/reviews_{str(sp).split('_')[-1]}.csv")
