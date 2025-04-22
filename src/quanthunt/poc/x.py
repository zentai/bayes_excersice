def turtle_trading(self):
    params = self.params
    base_df = self.base_df

    # 确定需要重新计算的索引
    idx_to_recalc = (
        base_df.index
        if base_df.ATR.isna().all()
        else base_df.ATR.iloc[params.ATR_sample :].isna().index
    )

    # 计算 turtle trading 指标
    base_df.loc[idx_to_recalc, "turtle_h"] = (
        base_df.Close.shift(1).rolling(params.upper_sample).max()
    )
    base_df.loc[idx_to_recalc, "turtle_l"] = (
        base_df.Close.shift(1).rolling(params.lower_sample).min()
    )

    # 计算其他相关指标
    base_df.loc[idx_to_recalc, "h_l"] = base_df.High - base_df.Low
    base_df.loc[idx_to_recalc, "c_h"] = (base_df.Close.shift(1) - base_df.High).abs()
    base_df.loc[idx_to_recalc, "c_l"] = (base_df.Close.shift(1) - base_df.Low).abs()
    base_df.loc[idx_to_recalc, "TR"] = base_df[["h_l", "c_h", "c_l"]].max(axis=1)
    base_df.loc[idx_to_recalc, "ATR"] = base_df["TR"].rolling(params.ATR_sample).mean()
    base_df.loc[idx_to_recalc, "Stop_profit"] = (
        base_df.Close.shift(1) - base_df.ATR.shift(1) * params.atr_loss_margin
    )
    base_df.loc[idx_to_recalc, "exit_price"] = base_df[["turtle_l", "Stop_profit"]].max(
        axis=1
    )

    return base_df
