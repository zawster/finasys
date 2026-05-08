# Portfolio API

Portfolio analytics operate on multi-symbol DataFrames with `timestamp`, `symbol`, and a price column such as `close`.

## `fs.portfolio.correlation_matrix(df, column="close", method="pearson")`

Returns a symbol-by-symbol return correlation matrix. `method` supports `"pearson"` and `"spearman"`.

## `fs.portfolio.covariance_matrix(df, column="close", annualize=True)`

Returns a symbol-by-symbol return covariance matrix. When `annualize=True`, values are multiplied by 252 trading days.

## `fs.portfolio.rolling_correlation(df, symbol_a, symbol_b, window=63, column="close")`

Returns pairwise rolling return correlation for two symbols.

## `fs.portfolio.portfolio_returns(df, weights, column="close")`

Returns weighted portfolio simple returns by timestamp.

```python
portfolio = fs.portfolio.portfolio_returns(df, {"AAPL": 0.5, "MSFT": 0.5})
```

## `fs.portfolio.equal_weight_returns(df, column="close")`

Returns equal-weight portfolio simple returns.

## `fs.portfolio.minimum_variance_weights(df, column="close")`

Returns minimum-variance weights as a `dict[str, float]`.
