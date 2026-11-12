#!/usr/bin/env python3
"""Run a Mode A PLS-PM mediation analysis and export a PDF report."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import landscape, letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from scipy.stats import norm


CONSTRUCTS = ("X", "M", "Y")
INDICATORS = {
    "X": tuple(f"X{i}" for i in range(1, 14)),
    "M": tuple(f"M{i}" for i in range(1, 10)),
    "Y": tuple(f"Y{i}" for i in range(1, 8)),
}
PATHS = (("X", "M"), ("X", "Y"), ("M", "Y"))


def _standardize(values: np.ndarray) -> np.ndarray:
    sd = values.std(axis=0, ddof=1)
    if np.any(sd == 0):
        raise ValueError("An indicator has no variation; PLS-SEM cannot be estimated.")
    return (values - values.mean(axis=0)) / sd


def _regression_coefficients(scores: dict[str, np.ndarray], predictors: tuple[str, ...],
                             outcome: str) -> np.ndarray:
    design = np.column_stack([scores[name] for name in predictors])
    return np.linalg.lstsq(design, scores[outcome], rcond=None)[0]


def fit_pls(data: np.ndarray) -> dict[str, Any]:
    """Estimate a reflective Mode A model using the path weighting scheme."""
    data = _standardize(data)
    offsets: dict[str, slice] = {}
    start = 0
    for construct in CONSTRUCTS:
        end = start + len(INDICATORS[construct])
        offsets[construct] = slice(start, end)
        start = end

    weights = {
        construct: np.full(len(INDICATORS[construct]), 1 / math.sqrt(len(INDICATORS[construct])))
        for construct in CONSTRUCTS
    }

    def make_scores(current_weights: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        scores: dict[str, np.ndarray] = {}
        for name in CONSTRUCTS:
            score = data[:, offsets[name]] @ current_weights[name]
            sd = score.std(ddof=1)
            if sd == 0:
                raise ValueError(f"Construct {name} has no score variation.")
            scores[name] = (score - score.mean()) / sd
        return scores

    for _ in range(500):
        scores = make_scores(weights)
        correlations = np.corrcoef(np.column_stack([scores[name] for name in CONSTRUCTS]), rowvar=False)
        score_index = {name: index for index, name in enumerate(CONSTRUCTS)}
        next_weights: dict[str, np.ndarray] = {}

        for construct in CONSTRUCTS:
            inner_estimate = np.zeros(data.shape[0])
            predecessors = tuple(source for source, target in PATHS if target == construct)
            successors = tuple(target for source, target in PATHS if source == construct)

            if predecessors:
                betas = _regression_coefficients(scores, predecessors, construct)
                for source, beta in zip(predecessors, betas):
                    inner_estimate += beta * scores[source]
            for target in successors:
                corr = correlations[score_index[construct], score_index[target]]
                inner_estimate += corr * scores[target]

            block = data[:, offsets[construct]]
            updated = block.T @ inner_estimate / (data.shape[0] - 1)
            norm_value = np.linalg.norm(updated)
            if norm_value == 0:
                raise ValueError(f"Outer weights for {construct} collapsed to zero.")
            next_weights[construct] = updated / norm_value

            if np.dot(next_weights[construct], weights[construct]) < 0:
                next_weights[construct] *= -1

        delta = max(
            np.max(np.abs(next_weights[name] - weights[name])) for name in CONSTRUCTS
        )
        weights = next_weights
        if delta < 1e-8:
            break
    else:
        raise RuntimeError("PLS algorithm did not converge after 500 iterations.")

    scores = make_scores(weights)
    loadings: dict[str, np.ndarray] = {}
    for name in CONSTRUCTS:
        loadings[name] = data[:, offsets[name]].T @ scores[name] / (data.shape[0] - 1)
        if loadings[name].mean() < 0:
            weights[name] *= -1
            scores[name] *= -1
            loadings[name] *= -1

    path_values: dict[tuple[str, str], float] = {}
    r_squared: dict[str, float] = {}
    for target in ("M", "Y"):
        predictors = tuple(source for source, destination in PATHS if destination == target)
        betas = _regression_coefficients(scores, predictors, target)
        for source, beta in zip(predictors, betas):
            path_values[(source, target)] = float(beta)
        residual = scores[target] - np.column_stack(
            [scores[source] for source in predictors]
        ) @ betas
        r_squared[target] = float(1 - np.var(residual, ddof=1) / np.var(scores[target], ddof=1))

    item_names = tuple(item for name in CONSTRUCTS for item in INDICATORS[name])
    score_matrix = np.column_stack([scores[name] for name in CONSTRUCTS])
    crossloadings = data.T @ score_matrix / (data.shape[0] - 1)
    latent_corr = np.corrcoef(score_matrix, rowvar=False)
    return {
        "data": data,
        "weights": weights,
        "scores": scores,
        "loadings": loadings,
        "paths": path_values,
        "r_squared": r_squared,
        "crossloadings": pd.DataFrame(crossloadings, index=item_names, columns=CONSTRUCTS),
        "latent_corr": pd.DataFrame(latent_corr, index=CONSTRUCTS, columns=CONSTRUCTS),
    }


def _reliability(items: np.ndarray, loadings: np.ndarray, weights: np.ndarray) -> dict[str, float]:
    item_corr = np.corrcoef(items, rowvar=False)
    count = items.shape[1]
    alpha = count / (count - 1) * (1 - np.trace(item_corr) / item_corr.sum())
    ave = float(np.mean(loadings ** 2))
    rho_c = float(np.sum(loadings) ** 2 / (
        np.sum(loadings) ** 2 + np.sum(1 - loadings ** 2)
    ))
    # Dijkstra-Henseler rho_A, with outer weights scaled to a unit-variance score.
    scaled_weights = weights / np.std(items @ weights, ddof=1)
    covariance = np.cov(items, rowvar=False)
    np.fill_diagonal(covariance, 0)
    weight_products = np.outer(scaled_weights, scaled_weights)
    np.fill_diagonal(weight_products, 0)
    denominator = float(scaled_weights @ weight_products @ scaled_weights)
    if denominator <= 0:
        raise ValueError("rho_A could not be calculated for this indicator block.")
    rho_a = float(
        np.dot(scaled_weights, scaled_weights) ** 2
        * (scaled_weights @ covariance @ scaled_weights)
        / denominator
    )
    return {"alpha": float(alpha), "rhoA": float(rho_a), "rhoC": rho_c, "AVE": ave}


def _htmt(data: np.ndarray) -> dict[tuple[str, str], float]:
    correlations = np.corrcoef(data, rowvar=False)
    indices: dict[str, np.ndarray] = {}
    start = 0
    for construct in CONSTRUCTS:
        end = start + len(INDICATORS[construct])
        indices[construct] = np.arange(start, end)
        start = end

    result: dict[tuple[str, str], float] = {}
    for left_index, left in enumerate(CONSTRUCTS):
        for right in CONSTRUCTS[left_index + 1:]:
            between = np.abs(correlations[np.ix_(indices[left], indices[right])]).mean()
            left_within = np.abs(correlations[np.ix_(indices[left], indices[left])])
            right_within = np.abs(correlations[np.ix_(indices[right], indices[right])])
            left_mean = left_within[np.triu_indices(len(indices[left]), k=1)].mean()
            right_mean = right_within[np.triu_indices(len(indices[right]), k=1)].mean()
            denominator = math.sqrt(float(left_mean * right_mean))
            result[(left, right)] = float(between / denominator)
    return result


def _effects(paths: dict[tuple[str, str], float]) -> dict[str, dict[str, float]]:
    direct_xy = paths[("X", "Y")]
    indirect_xy = paths[("X", "M")] * paths[("M", "Y")]
    return {
        "direct": {
            "X -> M": paths[("X", "M")],
            "X -> Y": direct_xy,
            "M -> Y": paths[("M", "Y")],
        },
        "indirect": {"X -> M -> Y": indirect_xy},
        "total": {"X -> Y": direct_xy + indirect_xy},
    }


def _effect_rows(original: dict[str, float], samples: dict[str, list[float]]) -> list[list[Any]]:
    rows = []
    for label, estimate in original.items():
        draws = np.asarray(samples[label], dtype=float)
        if len(draws) < 100:
            raise RuntimeError(
                f"Only {len(draws)} valid bootstrap samples for {label}; at least 100 are required."
            )
        sd = float(draws.std(ddof=1))
        t_value = float(estimate / sd) if sd else math.inf
        p_value = float(2 * norm.sf(abs(t_value)))
        rows.append([
            label, estimate, float(draws.mean()), sd, t_value, p_value,
            float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975)),
        ])
    return rows


def analyze(data: pd.DataFrame, bootstrap_count: int, seed: int) -> dict[str, Any]:
    item_names = [item for name in CONSTRUCTS for item in INDICATORS[name]]
    item_data = data[item_names].to_numpy(dtype=float)
    model = fit_pls(item_data)

    reliability: dict[str, dict[str, float]] = {}
    for name in CONSTRUCTS:
        start = sum(len(INDICATORS[c]) for c in CONSTRUCTS[:CONSTRUCTS.index(name)])
        block = model["data"][:, start:start + len(INDICATORS[name])]
        reliability[name] = _reliability(block, model["loadings"][name], model["weights"][name])

    complete_case_r2: dict[str, dict[str, float]] = {}
    for target in ("M", "Y"):
        predictors = tuple(source for source, destination in PATHS if destination == target)
        count = len(predictors)
        r2 = model["r_squared"][target]
        complete_case_r2[target] = {
            "R2": r2,
            "AdjR2": 1 - (1 - r2) * (len(data) - 1) / (len(data) - count - 1),
        }

    f_squared: dict[tuple[str, str], float] = {}
    for source, target in PATHS:
        predictors = tuple(p for p, destination in PATHS if destination == target)
        full_r2 = model["r_squared"][target]
        reduced_predictors = tuple(p for p in predictors if p != source)
        if not reduced_predictors:
            reduced_r2 = 0.0
        else:
            beta = _regression_coefficients(model["scores"], reduced_predictors, target)
            residual = model["scores"][target] - np.column_stack(
                [model["scores"][p] for p in reduced_predictors]
            ) @ beta
            reduced_r2 = float(1 - np.var(residual, ddof=1))
        f_squared[(source, target)] = (full_r2 - reduced_r2) / (1 - full_r2)

    vif: dict[str, float] = {}
    for target in ("M", "Y"):
        predictors = tuple(source for source, destination in PATHS if destination == target)
        for source in predictors:
            others = tuple(p for p in predictors if p != source)
            if not others:
                vif[f"{source} -> {target}"] = 1.0
            else:
                beta = _regression_coefficients(model["scores"], others, source)
                residual = model["scores"][source] - np.column_stack(
                    [model["scores"][p] for p in others]
                ) @ beta
                r2 = 1 - np.var(residual, ddof=1)
                vif[f"{source} -> {target}"] = 1 / (1 - r2)

    original_effects = _effects(model["paths"])
    boot_effects = {
        group: {label: [] for label in values}
        for group, values in original_effects.items()
    }
    boot_loadings = {name: {item: [] for item in INDICATORS[name]} for name in CONSTRUCTS}
    boot_htmt = {pair: [] for pair in _htmt(item_data)}
    boot_r2 = {name: [] for name in ("M", "Y")}
    rng = np.random.default_rng(seed)

    for iteration in range(bootstrap_count):
        indices = rng.integers(0, len(item_data), len(item_data))
        try:
            sample = fit_pls(item_data[indices])
            effects = _effects(sample["paths"])
            for group, values in effects.items():
                for label, estimate in values.items():
                    boot_effects[group][label].append(estimate)
            for name in CONSTRUCTS:
                for item, loading in zip(INDICATORS[name], sample["loadings"][name]):
                    boot_loadings[name][item].append(float(loading))
            for pair, value in _htmt(item_data[indices]).items():
                boot_htmt[pair].append(value)
            for name in ("M", "Y"):
                boot_r2[name].append(sample["r_squared"][name])
        except (ValueError, RuntimeError, np.linalg.LinAlgError):
            continue
        if (iteration + 1) % 500 == 0:
            print(f"Bootstrap: {iteration + 1}/{bootstrap_count}")

    valid_bootstrap = min(len(values) for group in boot_effects.values() for values in group.values())
    if valid_bootstrap < 100:
        raise RuntimeError(
            f"Only {valid_bootstrap} valid bootstrap iterations; cannot produce reliable intervals."
        )

    return {
        "model": model,
        "reliability": reliability,
        "r2": complete_case_r2,
        "f2": f_squared,
        "vif": vif,
        "effects": {
            group: _effect_rows(original_effects[group], boot_effects[group])
            for group in original_effects
        },
        "bootstrap_loadings": boot_loadings,
        "bootstrap_htmt": boot_htmt,
        "bootstrap_r2": boot_r2,
        "bootstrap_valid": valid_bootstrap,
        "htmt": _htmt(item_data),
        "n": len(data),
    }


def _format(value: Any) -> str:
    if isinstance(value, (float, np.floating)):
        return f"{value:.4f}"
    return str(value)


def _add_table(story: list[Any], title: str, headers: list[str], rows: list[list[Any]]) -> None:
    styles = getSampleStyleSheet()
    story.append(Paragraph(title, styles["Heading2"]))
    data = [[Paragraph(str(cell), styles["BodyText"]) for cell in headers]]
    data.extend([[_format(cell) for cell in row] for row in rows])
    widths = [None] * len(headers)
    table = Table(data, repeatRows=1, colWidths=widths, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#24476b")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 7.5),
        ("LEADING", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#b7c5d3")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#edf3f8")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.extend([table, Spacer(1, 12)])


def write_report(results: dict[str, Any], output: Path, bootstrap_count: int, seed: int) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(
        str(output), pagesize=landscape(letter),
        leftMargin=0.45 * inch, rightMargin=0.45 * inch,
        topMargin=0.55 * inch, bottomMargin=0.5 * inch,
        title="PLS-SEM Results",
    )
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="ReportTitle", parent=styles["Title"], alignment=TA_CENTER,
        textColor=colors.HexColor("#24476b"), spaceAfter=8,
    ))
    story: list[Any] = [
        Paragraph("PLS-SEM Results", styles["ReportTitle"]),
        Paragraph(
            f"Respondents: {results['n']} | Indicators: 29 | "
            f"Bootstrap samples requested: {bootstrap_count} "
            f"(valid: {results['bootstrap_valid']})",
            styles["BodyText"],
        ),
        Paragraph(
            "Model: X -> M -> Y, with a direct X -> Y path. "
            "Estimator: PLS-PM, Mode A, path weighting.",
            styles["BodyText"],
        ),
        Paragraph(
            f"Confidence intervals: percentile bootstrap, 95%. "
            f"P values are two-sided normal approximations. Seed: {seed}.",
            styles["BodyText"],
        ),
        Paragraph(
            "Indicators with loadings >= 0.708 meet the common guideline; "
            "lower loadings are flagged for review alongside AVE, reliability, "
            "and theory. These heuristics do not by themselves establish model validity.",
            styles["BodyText"],
        ),
        Spacer(1, 12),
    ]

    model = results["model"]
    _add_table(story, "01 Outer loadings",
               ["Construct", "Indicator", "Loading", "Evaluation"], [
        [name, item, float(loading),
         "Meets loading guideline" if loading >= 0.708 else "Review with AVE, reliability, and theory"]
        for name in CONSTRUCTS
        for item, loading in zip(INDICATORS[name], model["loadings"][name])
    ])
    _add_table(story, "02 Reliability AVE", ["Term", "alpha", "rhoA", "rhoC", "AVE"], [
        [name, *[results["reliability"][name][key] for key in ("alpha", "rhoA", "rhoC", "AVE")]]
        for name in CONSTRUCTS
    ])

    htmt = results["htmt"]
    _add_table(story, "03 HTMT", ["Term", "X", "M", "Y"], [
        ["X", "", "", ""],
        ["M", htmt[("X", "M")], "", ""],
        ["Y", htmt[("X", "Y")], htmt[("M", "Y")], ""],
    ])
    ave = {name: results["reliability"][name]["AVE"] for name in CONSTRUCTS}
    corr = model["latent_corr"]
    _add_table(story, "04 Fornell Larcker", ["Term", *CONSTRUCTS], [
        [left, *[
            math.sqrt(ave[left]) if left == right else float(corr.loc[left, right])
            for right in CONSTRUCTS
        ]]
        for left in CONSTRUCTS
    ])
    _add_table(story, "05 Cross loadings", ["Term", *CONSTRUCTS], [
        [item, *[float(model["crossloadings"].loc[item, name]) for name in CONSTRUCTS]]
        for item in [item for name in CONSTRUCTS for item in INDICATORS[name]]
    ])
    _add_table(story, "06 R square", ["Term", "M", "Y"], [
        ["R^2", results["r2"]["M"]["R2"], results["r2"]["Y"]["R2"]],
        ["AdjR^2", results["r2"]["M"]["AdjR2"], results["r2"]["Y"]["AdjR2"]],
    ])

    _add_table(story, "07 F square", ["Term", "X", "M", "Y"], [
        [source, *[
            results["f2"].get((source, target), 0.0)
            for target in CONSTRUCTS
        ]]
        for source in CONSTRUCTS
    ])
    _add_table(story, "08 Inner VIF", ["Predictor -> outcome", "VIF"], [
        [label, value] for label, value in results["vif"].items()
    ])
    effect_headers = [
        "Effect", "Original_O", "Mean_M", "STDEV", "T_statistic",
        "P_normal_approx", "CI_2.5", "CI_97.5",
    ]
    for section, title in (
        ("direct", "09 Direct effects"),
        ("indirect", "10 Indirect effects"),
        ("total", "11 Total effects"),
    ):
        _add_table(story, title, effect_headers, results["effects"][section])

    loading_rows = []
    for name in CONSTRUCTS:
        for item in INDICATORS[name]:
            draws = np.asarray(results["bootstrap_loadings"][name][item])
            estimate = float(model["loadings"][name][INDICATORS[name].index(item)])
            sd = float(draws.std(ddof=1))
            loading_rows.append([
                f"{item} -> {name}", estimate, float(draws.mean()), sd,
                estimate / sd if sd else math.inf,
                float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975)),
            ])
    _add_table(story, "12 Bootstrap loadings", [
        "Term", "Original Est.", "Bootstrap Mean", "Bootstrap SD",
        "T Stat.", "2.5% CI", "97.5% CI",
    ], loading_rows)

    htmt_rows = []
    for pair, estimate in htmt.items():
        draws = np.asarray(results["bootstrap_htmt"][pair])
        sd = float(draws.std(ddof=1))
        htmt_rows.append([
            f"{pair[0]} -> {pair[1]}", estimate, float(draws.mean()), sd,
            estimate / sd if sd else math.inf,
            float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975)),
        ])
    _add_table(story, "13 Bootstrap HTMT", [
        "Term", "Original Est.", "Bootstrap Mean", "Bootstrap SD",
        "T Stat.", "2.5% CI", "97.5% CI",
    ], htmt_rows)

    def page_footer(canvas: Any, doc: Any) -> None:
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#667788"))
        canvas.drawString(0.45 * inch, 0.25 * inch, "PLS-SEM analysis")
        canvas.drawRightString(landscape(letter)[0] - 0.45 * inch, 0.25 * inch, f"{doc.page}")
        canvas.restoreState()

    document.build(story, onFirstPage=page_footer, onLaterPages=page_footer)


def load_responses(path: Path) -> tuple[pd.DataFrame, int]:
    if not path.is_file():
        raise FileNotFoundError(f"Excel workbook not found: {path}")
    frame = pd.read_excel(path, header=1, engine="openpyxl")
    indicators = [item for construct in CONSTRUCTS for item in INDICATORS[construct]]
    missing_columns = [item for item in indicators if item not in frame.columns]
    if missing_columns:
        raise ValueError(
            "Expected indicator codes were not found in the second Excel header row: "
            + ", ".join(missing_columns)
        )
    raw = frame[indicators].apply(pd.to_numeric, errors="coerce")
    complete = raw.notna().all(axis=1)
    excluded = int((~complete).sum())
    clean = raw.loc[complete].astype(float)
    if clean.empty:
        raise ValueError("No complete responses were found for all 29 indicators.")
    if not clean.isin([1, 2, 3, 4, 5]).all().all():
        raise ValueError("All indicator responses must be Likert values from 1 to 5.")
    return clean.reset_index(drop=True), excluded


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Path to the questionnaire .xlsx file")
    parser.add_argument(
        "--output", type=Path, default=Path("PLS-SEM Results Python.pdf"),
        help="Output PDF path (default: PLS-SEM Results Python.pdf)",
    )
    parser.add_argument("--bootstrap", type=int, default=5000, help="Bootstrap resamples (default: 5000)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()
    if args.bootstrap < 100:
        parser.error("--bootstrap must be at least 100")

    responses, excluded = load_responses(args.input)
    print(f"Complete responses: {len(responses)}; excluded incomplete responses: {excluded}")
    results = analyze(responses, args.bootstrap, args.seed)
    write_report(results, args.output, args.bootstrap, args.seed)
    print(f"Report saved: {args.output.resolve()}")


if __name__ == "__main__":
    main()
