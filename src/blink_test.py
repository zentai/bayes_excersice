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
    stock_list = "0018.KL,1023.KL,3026.KL,4677.KL,5099.KL,5200.KL,5299.KL,7033.KL,8567.KL,0023.KL,1066.KL,3336.KL,4707.KL,5109.KL,5202.KL,5318.KL,7084.KL,8583.KL,0041.KL,1155.KL,3417.KL,4715.KL,5148.KL,5204.KL,5347.KL,7106.KL,8869.KL,0072.KL,1171.KL,3476.KL,5008.KL,5168.KL,5205.KL,5819.KL,7113.KL,0078.KL,1295.KL,3689.KL,5014.KL,5175.KL,5209.KL,6012.KL,7123.KL,0082.KL,1562.KL,3743.KL,5020.KL,5182.KL,5239.KL,6076.KL,7153.KL,0138.KL,1651.KL,3794.KL,5038.KL,5183.KL,5246.KL,6556.KL,7164.KL,0146.KL,1724.KL,3816.KL,5053.KL,5184.KL,5264.KL,6742.KL,7183.KL,0166.KL,2445.KL,3867.KL,5062.KL,5185.KL,5279.KL,6888.KL,7216.KL,0216.KL,2828.KL,4065.KL,5075.KL,5196.KL,5285.KL,6947.KL,7293.KL".split(
        ","
    )

    # Initialize an empty list to collect reviews
    reviews = []

    # Set up ProcessPoolExecutor with a fixed number of workers (based on CPU cores)
    max_workers = (
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
        df_reviews.to_csv(f"{REPORTS_DIR}/df_reviews.csv", index=False)
        print(df_reviews)
        print(f"created: {REPORTS_DIR}/df_reviews.csv")
