"""Used-car sale-price model for the auction buy/pass decision.

Fit on 70% of used_car_sales.csv. The held-out 30% is only for fit.
auction_lots.csv is only for the buy/pass rule and is never used to fit.

Running this file prints every figure used on the slides and writes
predicted_vs_actual.png next to this script. That chart overlays the
baseline and v1 on the same axes. Rounded slide figures are computed
here; they are not typed in by hand.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

DATA_DIR = Path(__file__).resolve().parent
PREP_COST = 1500
RANDOM_STATE = 42
TEST_SIZE = 0.30

ACCIDENT = "C(Accident, Treatment(reference='no'))[T.yes]"
FULL_FORMULA = (
    "Sale_Price ~ Mileage_K + Age_Years + Horsepower"
    " + C(Accident, Treatment(reference='no')) + Num_Owners"
)
V1_FORMULA = (
    "Sale_Price ~ Mileage_K + Age_Years + Horsepower"
    " + C(Accident, Treatment(reference='no'))"
)
DROP_MILEAGE_FORMULA = (
    "Sale_Price ~ Age_Years + Horsepower"
    " + C(Accident, Treatment(reference='no')) + Num_Owners"
)
DROP_AGE_FORMULA = (
    "Sale_Price ~ Mileage_K + Horsepower"
    " + C(Accident, Treatment(reference='no')) + Num_Owners"
)


def decide(predicted_sale_price, auction_price, prep_cost=PREP_COST):
    """Buy when the predicted lot price clears the auction price plus prep cost."""
    if predicted_sale_price > auction_price + prep_cost:
        return "buy"
    return "pass"


def profit_if_bought(later_sale_price, auction_price, prep_cost=PREP_COST):
    return later_sale_price - auction_price - prep_cost


def rule_summary(lots, decision_column):
    bought = lots[lots[decision_column] == "buy"]
    losers = int((bought["profit_if_bought"] < 0).sum())
    return {
        "bought": int(len(bought)),
        "losers": losers,
        "total_profit": float(bought["profit_if_bought"].sum()),
    }


def money(value, decimals=0):
    """Format a dollar amount with the sign in front of the dollar sign."""
    amount = f"{abs(float(value)):,.{decimals}f}"
    if float(value) < 0:
        return f"−${amount}"
    return f"${amount}"


def nearest_10(value):
    return int(round(float(value) / 10.0) * 10)


def nearest_100(value):
    return int(round(float(value) / 100.0) * 100)


def signed_term(coef, name):
    sign = "−" if coef < 0 else "+"
    return f" {sign} {abs(coef):,.2f} × {name}"


def prediction_equation(v1):
    intercept = v1.params["Intercept"]
    return (
        f"{intercept:,.2f}"
        + signed_term(v1.params["Mileage_K"], "Mileage_K")
        + signed_term(v1.params["Age_Years"], "Age_Years")
        + signed_term(v1.params["Horsepower"], "Horsepower")
        + signed_term(v1.params[ACCIDENT], "accident_yes")
    )


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def prediction_limits(*series):
    """Axis limits covering every actual price and both sets of predictions."""
    values = np.concatenate([np.asarray(series_values, dtype=float).ravel() for series_values in series])
    low = float(values.min())
    high = float(values.max())
    pad = (high - low) * 0.06
    return low - pad, high + pad


def save_predicted_vs_actual(y_true, baseline_pred, v1_pred, path):
    """Save one chart with the baseline and v1 overlaid on the test cars."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    y_true = np.asarray(y_true, dtype=float)
    baseline_pred = np.asarray(baseline_pred, dtype=float)
    v1_pred = np.asarray(v1_pred, dtype=float)
    lo, hi = prediction_limits(y_true, baseline_pred, v1_pred)

    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    ax.scatter(
        y_true,
        baseline_pred,
        s=28,
        color="#1d4ed8",
        zorder=2,
        label=(
            f"Baseline  R² {r2_score(y_true, baseline_pred):.3f}"
            f"  typical error {money(rmse(y_true, baseline_pred))}"
        ),
    )
    ax.scatter(
        y_true,
        v1_pred,
        s=28,
        color="#c2410c",
        zorder=3,
        label=(
            f"v1  R² {r2_score(y_true, v1_pred):.3f}"
            f"  typical error {money(rmse(y_true, v1_pred))}"
        ),
    )
    ax.plot([lo, hi], [lo, hi], linestyle="--", color="#57534e", linewidth=1, label="Exact", zorder=1)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Actual sale price")
    ax.set_ylabel("Predicted sale price")
    ax.set_title(f"Predicted vs actual sale price, {len(y_true)} test cars")
    ax.legend(frameon=False, loc="upper left")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda value, _pos: f"${value / 1000:.0f}k"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _pos: f"${value / 1000:.0f}k"))
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def print_slide_numbers(sales, train, test, full, v1, no_mileage, no_age, scored):
    """Print each slide figure from the fitted objects."""
    owners = "Num_Owners"
    owners_ci = full.conf_int().loc[owners]
    corr = sales[["Mileage_K", "Age_Years", "Num_Owners"]].corr()

    y_test = test["Sale_Price"].to_numpy(dtype=float)
    baseline_price = float(train["Sale_Price"].mean())
    baseline_pred = np.full_like(y_test, baseline_price)
    full_pred = full.predict(test).to_numpy(dtype=float)
    v1_pred = v1.predict(test).to_numpy(dtype=float)

    clean = {"Mileage_K": 50, "Age_Years": 4, "Horsepower": 200, "Accident": "no"}
    accident = {**clean, "Accident": "yes"}
    clean_price = float(v1.predict(pd.DataFrame([clean])).iloc[0])
    accident_price = float(v1.predict(pd.DataFrame([accident])).iloc[0])

    baseline = rule_summary(scored, "rule_baseline")
    model = rule_summary(scored, "rule_v1")
    only_baseline = scored[
        (scored["rule_baseline"] == "buy") & (scored["rule_v1"] == "pass")
    ]
    only_v1 = scored[
        (scored["rule_v1"] == "buy") & (scored["rule_baseline"] == "pass")
    ]
    v1_only_winners = int((only_v1["profit_if_bought"] > 0).sum())
    lot_31 = scored.loc[scored["Lot"] == 31, "profit_if_bought"].iloc[0]

    print("SLIDE NUMBERS")
    print("These figures are calculated in this run.")
    print()
    print(f"Training cars: {len(train)}")
    print(f"Test cars: {len(test)}")
    print(f"random_state: {RANDOM_STATE}")
    print(f"Prep cost: {money(PREP_COST)}")
    print()
    print(
        f"Baseline price (training average): {money(baseline_price, 2)}"
        f"  →  {money(baseline_price)}"
    )
    print(
        f"Test-set average sale price: {money(test['Sale_Price'].mean(), 2)}"
        f"  →  {money(test['Sale_Price'].mean())}"
    )
    print()
    print("Num_Owners does not stay in v1")
    print(
        f"  coefficient: {money(full.params[owners], 2)} per owner"
        f"  →  {money(full.params[owners])}"
    )
    print(f"  p-value: {full.pvalues[owners]:.3f}")
    print(
        f"  95% CI: {money(owners_ci[0], 2)} to {money(owners_ci[1], 2)}"
        f"  →  {money(nearest_10(owners_ci[0]))} to {money(nearest_10(owners_ci[1]))}"
    )
    print(
        "  correlation on all 240 sales (description only):"
        f"  owners–mileage {corr.loc['Num_Owners', 'Mileage_K']:.2f},"
        f"  owners–age {corr.loc['Num_Owners', 'Age_Years']:.2f},"
        f"  mileage–age {corr.loc['Mileage_K', 'Age_Years']:.2f}"
    )
    print(
        f"  adjusted R² with owners: {full.rsquared_adj:.4f}"
        f"  →  {full.rsquared_adj:.3f}"
    )
    print(
        f"  adjusted R² v1: {v1.rsquared_adj:.4f}"
        f"  →  {v1.rsquared_adj:.3f}"
    )
    print(
        f"  training R² with owners: {full.rsquared:.4f}"
        f"  →  {full.rsquared:.3f}"
    )
    print(
        f"  training R² v1: {v1.rsquared:.4f}"
        f"  →  {v1.rsquared:.3f}"
    )
    print(
        f"  adjusted R² if mileage is dropped from the five-feature model:"
        f" {no_mileage.rsquared_adj:.4f}  →  {no_mileage.rsquared_adj:.3f}"
    )
    print(
        f"  adjusted R² if age is dropped from the five-feature model:"
        f" {no_age.rsquared_adj:.4f}  →  {no_age.rsquared_adj:.3f}"
    )
    print()
    print("v1 coefficients, holding the other features fixed")
    print(
        f"  intercept: {money(v1.params['Intercept'], 2)}"
        f"  →  {money(v1.params['Intercept'])}"
    )
    print(
        f"  mileage: {money(v1.params['Mileage_K'], 2)} per 1,000 miles"
        f"  →  {money(v1.params['Mileage_K'])}"
    )
    mileage_dollars = round(v1.params["Mileage_K"])
    print(
        f"  10,000 more miles: {money(10 * v1.params['Mileage_K'], 2)}"
        f"  →  {money(10 * mileage_dollars)} from the rounded per-thousand coefficient"
    )
    print(
        f"  age: {money(v1.params['Age_Years'], 2)} per year"
        f"  →  {money(v1.params['Age_Years'])}"
    )
    print(
        f"  horsepower: {money(v1.params['Horsepower'], 2)} per hp"
        f"  →  {money(v1.params['Horsepower'])}"
    )
    horsepower_dollars = round(v1.params["Horsepower"])
    print(
        f"  20 more horsepower: {money(20 * v1.params['Horsepower'], 2)}"
        f"  →  {money(20 * horsepower_dollars)} from the rounded per-hp coefficient"
    )
    print(
        f"  accident = yes: {money(v1.params[ACCIDENT], 2)}"
        f"  →  {money(v1.params[ACCIDENT])}"
    )
    slope_pvalues = v1.pvalues.drop(labels=["Intercept"])
    print("  p-values: " + ", ".join(f"{name} {p:.4g}" for name, p in slope_pvalues.items()))
    print(f"  largest slope p-value: {slope_pvalues.max():.4g}")
    print()
    print("v1 equation")
    print(f"  predicted price = {prediction_equation(v1)}")
    print()
    print("Example car: 50,000 miles, 4 years, 200 hp")
    print(
        f"  no accident: {money(clean_price, 2)}"
        f"  →  about {money(nearest_100(clean_price))}"
    )
    print(
        f"  accident: {money(accident_price, 2)}"
        f"  →  about {money(nearest_100(accident_price))}"
    )
    print()
    print("Test cars the model never saw")
    print(
        f"  baseline  R²={r2_score(y_test, baseline_pred):.4f}"
        f"  →  {r2_score(y_test, baseline_pred):.3f}"
        f"    RMSE={money(rmse(y_test, baseline_pred))}"
    )
    print(
        f"  v1        R²={r2_score(y_test, v1_pred):.4f}"
        f"  →  {r2_score(y_test, v1_pred):.3f}"
        f"    RMSE={money(rmse(y_test, v1_pred))}"
    )
    print(
        f"  five-feature  R²={r2_score(y_test, full_pred):.4f}"
        f"  →  {r2_score(y_test, full_pred):.3f}"
        f"    RMSE={money(rmse(y_test, full_pred))}"
    )
    print()
    print(f"Auction lots: {len(scored)}. Profit is counted only on cars the rule buys.")
    print(
        f"  baseline  bought {baseline['bought']}"
        f"  losers {baseline['losers']}"
        f"  total profit {money(baseline['total_profit'])}"
    )
    print(
        f"  v1        bought {model['bought']}"
        f"  losers {model['losers']}"
        f"  total profit {money(model['total_profit'])}"
    )
    print(
        f"  profit gap (v1 − baseline):"
        f" {money(model['total_profit'] - baseline['total_profit'])}"
    )
    print(
        f"  rules differ on {len(only_baseline) + len(only_v1)} lots"
    )
    print(
        f"  baseline buys and v1 passes: {len(only_baseline)} cars,"
        f" lots {only_baseline['Lot'].tolist()},"
        f" profit {money(only_baseline['profit_if_bought'].sum())}"
    )
    print(
        f"  v1 buys and baseline passes: {len(only_v1)} cars,"
        f" lots {only_v1['Lot'].tolist()},"
        f" profit {money(only_v1['profit_if_bought'].sum())},"
        f" of which {v1_only_winners} make money"
    )
    print(f"  lot 31 profit if bought: {money(lot_31)}")


def main():
    sales = pd.read_csv(DATA_DIR / "used_car_sales.csv")
    lots = pd.read_csv(DATA_DIR / "auction_lots.csv")

    train, test = train_test_split(
        sales, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    full = smf.ols(FULL_FORMULA, data=train).fit()
    v1 = smf.ols(V1_FORMULA, data=train).fit()
    no_mileage = smf.ols(DROP_MILEAGE_FORMULA, data=train).fit()
    no_age = smf.ols(DROP_AGE_FORMULA, data=train).fit()

    scored = lots.copy()
    baseline_price = float(train["Sale_Price"].mean())
    scored["pred_v1"] = v1.predict(scored)
    scored["pred_baseline"] = baseline_price
    scored["rule_v1"] = [
        decide(predicted, auction)
        for predicted, auction in zip(scored["pred_v1"], scored["Auction_Price"])
    ]
    scored["rule_baseline"] = [
        decide(predicted, auction)
        for predicted, auction in zip(scored["pred_baseline"], scored["Auction_Price"])
    ]
    scored["profit_if_bought"] = profit_if_bought(
        scored["Later_Sale_Price"], scored["Auction_Price"]
    )

    print_slide_numbers(sales, train, test, full, v1, no_mileage, no_age, scored)
    y_test = test["Sale_Price"].to_numpy(dtype=float)
    chart_path = save_predicted_vs_actual(
        y_test,
        np.full_like(y_test, float(train["Sale_Price"].mean())),
        v1.predict(test).to_numpy(dtype=float),
        DATA_DIR / "predicted_vs_actual.png",
    )
    print()
    print(f"Chart saved to {chart_path.name}")
    print()
    print("FIVE-FEATURE MODEL (includes Num_Owners)")
    print(full.summary())
    print()
    print("v1 (Num_Owners dropped)")
    print(v1.summary())

    return v1


if __name__ == "__main__":
    main()
