"""WildChat data loading, preprocessing, and metrics container."""

from dataclasses import dataclass, replace
import pickle
import polars as pl
import numpy as np
import datetime
import os

# ── Constants ──────────────────────────────────────────────────────────────────

OUTPUT_DIR = "outputs"  # base directory for all generated figures, tables, and CSVs
CUTOFF = datetime.date(2024, 9, 1)
VLINE_DATE = np.datetime64('2024-09-01')
VERSIONS = [(f"{OUTPUT_DIR}/wildchat-full", True, ""), (f"{OUTPUT_DIR}/wildchat-trunc", False, "_trunc")]


# ── Dataset registry ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class DatasetConfig:
    """Per-dataset paths/shape and (optional) trunc cutoff.

    cutoff=None disables the truncated activity-group version and skips
    before/after columns in template metrics.
    """
    name: str
    source_data_dir: str
    data_dir: str
    file_prefix: str        # e.g. "wildchat-4.8m"
    n_shards: int
    cache_path: str
    cutoff: datetime.date | None


DATASETS = {
    "wildchat-4.8m": DatasetConfig(
        name="wildchat-4.8m",
        source_data_dir="wildchat-data/wildchat-4.8m",
        data_dir="wildchat-results/wildchat-4.8m-results",
        file_prefix="wildchat-4.8m",
        n_shards=86,
        cache_path="wildchat_metrics_cache.pkl",
        cutoff=CUTOFF,
    ),
    "wildchat-1m": DatasetConfig(
        name="wildchat-1m",
        source_data_dir="wildchat-data/wildchat-1m",
        data_dir="wildchat-results/wildchat-1m-results",
        file_prefix="wildchat-1m",
        n_shards=14,
        cache_path="wildchat_1m_metrics_cache.pkl",
        cutoff=None,
    ),
}

wildchat_color = '#d94f3d'
model_colors = {
    'gpt-3.5-turbo-0125': '#5cd6d6', 'gpt-3.5-turbo-0301': '#2eb8b8',
    'gpt-3.5-turbo-0613': '#1f7a7a', 'gpt-4-0125-preview': '#c56db3',
    'gpt-4-0314': '#b649a0', 'gpt-4-0613': '#923a80',
    'gpt-4-1106-preview': '#6d2c60', 'gpt-4-turbo-2024-04-09': '#08415C',
    'gpt-4.1-mini-2025-04-14': '#FCDC4D', 'gpt-4o-2024-05-13': '#d5e68d',
    'gpt-4o-2024-08-06': '#b3d12e', 'gpt-4o-2024-11-20': '#7d9220',
    'gpt-4o-mini-2024-07-18': '#FEB95F', 'o1-mini-2024-09-12': '#DCD6F7',
    'o1-preview-2024-09-12': '#FF5A5F',
}

intent_map = {
    'ANALYSIS': 'Analysis', 'IMAGE_CREATION': 'Image Creation',
    'INFORMATION_GATHERING': 'Information Gathering',
    'INFORMATION_LOOKUP': 'Information Lookup',
    'OPEN_ENDED_DISCOVERY': 'Open-Ended Discovery',
    'SUMMARIZATION': 'Summarization', 'TEXT_GENERATION': 'Text Generation',
    'TRANSLATION_OR_CONVERSION': 'Translation or Conversion',
    'WEB_SITE_NAVIGATION': 'Website Navigation',
}

domain_map = {
    'ADULT': 'Adult', 'BIOLOGY': 'Biology',
    'BUSINESS_AND_FINANCE': 'Business and Finance',
    'COMPUTERS_AND_ELECTRONICS': 'Computers and Electronics',
    'CREATIVE_WRITING_AND_EDITING': 'Creative Writing and Editing',
    'DATA_ANALYSIS_AND_VISUALIZATION': 'Data Analysis and Visualization',
    'EDUCATION_AND_LEARNING': 'Education and Learning',
    'ENGINEERING_AND_DESIGN': 'Engineering and Design',
    'ENTERTAINMENT': 'Entertainment', 'FASHION_AND_BEAUTY': 'Fashion and Beauty',
    'FOOD_AND_DRINK': 'Food and Drink', 'GAMING': 'Gaming',
    'HEALTH_AND_MEDICINE': 'Health and Medicine',
    'HISTORY_AND_CULTURE': 'History and Culture',
    'HOME_AND_AUTO': 'Home and Auto',
    'JOBS_AND_EMPLOYMENT': 'Jobs and Employment',
    'LAW_AND_POLITICS': 'Law and Politics',
    'MACHINE_LEARNING_AND_AI': 'Machine Learning and AI',
    'MARKETING_AND_SALES': 'Marketing and Sales',
    'MATHEMATICS_AND_LOGIC': 'Mathematics and Logic', 'OTHER': 'Other',
    'PHYSICS_AND_CHEMISTRY': 'Physics and Chemistry',
    'PROFESSIONAL_WRITING_AND_EDITING': 'Professional Writing and Editing',
    'PROGRAMMING_AND_SCRIPTING': 'Programming and Scripting',
    'RELIGION_AND_PHILOSOPHY': 'Religion and Philosophy',
    'SHOPPING_AND_ECOMMERCE': 'Shopping and eCommerce',
    'SMALL_TALK_AND_CHATBOT': 'Small Talk and Chatbot',
    'SPORTS_AND_FITNESS': 'Sports and Fitness',
    'TRANSLATION_AND_LANGUAGE': 'Translation and Language',
    'TRAVEL_AND_TOURISM': 'Travel and Tourism',
}

levels = ['low', 'middle', 'high']

# ── Helpers ────────────────────────────────────────────────────────────────────

def trunc(df, date_col="date"):
    return df.filter(pl.col(date_col) < CUTOFF)


def make_dummies(df, col, prefix):
    df = df.with_columns(pl.col(col).str.replace(prefix, "").alias(col))
    dummies = df[col].to_dummies().rename(lambda c: c.replace(f"{col}_", ""))
    return df.with_columns(dummies)


# ── Metrics Container ─────────────────────────────────────────────────────────

@dataclass
class WildChatMetrics:
    intent_cols: list
    domain_cols: list

    # Time series aggregations (pre-activity-group filter)
    conv_by_day_df: pl.DataFrame
    user_by_day_df: pl.DataFrame
    mssgs_by_day_df: pl.DataFrame
    complete_day_df: pl.DataFrame
    repeat_by_day_df: pl.DataFrame
    sent_len_by_day_df: pl.DataFrame
    intent_day_mean_df: pl.DataFrame
    domain_day_mean_df: pl.DataFrame

    # By-model aggregations (pre-activity-group filter)
    conv_by_model_day_df: pl.DataFrame
    user_by_model_day_df: pl.DataFrame
    complete_model_day_df: pl.DataFrame
    intent_model_day_df: pl.DataFrame
    domain_model_day_df: pl.DataFrame

    # Full filtered DataFrame (with intent/domain dummies, pre-activity-group)
    filt_df_all: pl.DataFrame

    # Post-activity-group: [(dir_name, df, tsuffix)]
    ag_versions: list

    # Scalars
    mean_sentence_len: float
    completion_rate_overall: float

    # Dataset config (name, paths, cutoff)
    dataset: DatasetConfig = None

    # Templated / API-like usage detection
    template_by_day_df: pl.DataFrame = None
    template_groups_df: pl.DataFrame = None  # fingerprint + count, sorted desc
    template_daily_by_fp_df: pl.DataFrame = None  # fingerprint + date + count


# ── Loader ─────────────────────────────────────────────────────────────────────

def _result_paths(data_dir, prefix, n_shards, result_type):
    paths = []
    for i in range(n_shards):
        filename = f"{prefix}-{i:05d}-{result_type}.parquet"
        direct_path = f"{data_dir}/{filename}"
        stripped_path = f"{data_dir}/no-content/no-content-{filename}"
        if os.path.exists(direct_path):
            paths.append(direct_path)
        elif os.path.exists(stripped_path):
            paths.append(stripped_path)
        else:
            raise FileNotFoundError(f"Could not find {direct_path} or {stripped_path}")
    return paths


def _source_paths(cfg):
    return [
        f"{cfg.source_data_dir}/train-{i:05d}-of-{cfg.n_shards:05d}.parquet"
        for i in range(cfg.n_shards)
    ]


def load_source_conversations(cfg):
    """Load source conversation keys and full, unabridged conversations."""
    key_cols = ["conversation_hash", "model", "timestamp", "turn", "language", "hashed_ip", "state", "country"]
    return (
        pl.scan_parquet(_source_paths(cfg))
        .select(key_cols + ["conversation"])
        .collect()
        .unique(subset=key_cols)
    )


def load_source_text(cfg):
    """Recreate the abbreviated Text representation used in the paper."""
    content = pl.element().struct.field("content")
    abbreviated_content = (
        pl.when(content.str.len_chars() < 5000)
        .then(content)
        .otherwise(
            content.str.slice(0, 2500)
            + pl.lit(" [... more text ...] ")
            + content.str.slice(-2500)
        )
    )
    message = pl.concat_str(
        pl.lit("<| start "),
        pl.element().struct.field("role").replace({"assistant": "agent"}),
        pl.lit(" message |>\n"),
        abbreviated_content,
        pl.lit("\n<| end "),
        pl.element().struct.field("role").replace({"assistant": "agent"}),
        pl.lit(" message |>")
    )
    return (
        load_source_conversations(cfg)
        .with_columns(pl.col("conversation").list.eval(message).list.join("\n\n").alias("Text"))
        .drop("conversation")
    )


def load_metrics(dataset="wildchat-4.8m", data_dir=None, cache_path=None, use_cache=True):
    """Load and preprocess all WildChat data. Results are cached to disk.

    Args:
        dataset: Key into DATASETS (e.g. "wildchat-4.8m", "wildchat-1m") or a
            DatasetConfig instance.
        data_dir: Override the dataset's data_dir (optional).
        cache_path: Override the dataset's cache_path (optional). Set explicitly
            to None-via-empty-string "" to disable caching.
        use_cache: If True and cache exists, load from cache.
    """
    cfg = dataset if isinstance(dataset, DatasetConfig) else DATASETS[dataset]
    if data_dir is not None:
        cfg = replace(cfg, data_dir=data_dir)
    if cache_path == "":
        cache_path = None
    elif cache_path is None:
        cache_path = cfg.cache_path

    os.makedirs(f"{OUTPUT_DIR}/wildchat-full", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/wildchat-trunc", exist_ok=True)

    if use_cache and cache_path and os.path.exists(cache_path):
        print(f"Loading cached metrics from {cache_path}...")
        with open(cache_path, "rb") as f:
            m = pickle.load(f)
        print("Cached metrics loaded.")
        return m

    data_dir = cfg.data_dir
    prefix = cfg.file_prefix
    n = cfg.n_shards

    # ── Load raw parquet files ──
    _key_cols = ['conversation_hash', 'model', 'timestamp', 'turn', 'language', 'hashed_ip', 'state', 'country']
    print("Loading domain data...")
    domain_raw = (
        pl.scan_parquet(_result_paths(data_dir, prefix, n, "domain"))
        .select(_key_cols + ['conversation_domain'])
        .collect()
    )
    print("Loading intent data...")
    intent_raw = (
        pl.scan_parquet(_result_paths(data_dir, prefix, n, "intent"), missing_columns='insert')
        .select('user_intent')
        .collect()
    )
    print("Loading complete data...")
    complete_raw = (
        pl.scan_parquet(_result_paths(data_dir, prefix, n, "task"), missing_columns='insert')
        .select('completed')
        .collect()
    )

    # ── Merge (parquets are co-sharded: same rows, same order) ──
    assert len(domain_raw) == len(intent_raw) == len(complete_raw), (
        f"Row count mismatch: domain={len(domain_raw)}, intent={len(intent_raw)}, complete={len(complete_raw)}"
    )
    df = domain_raw.hstack(intent_raw).hstack(complete_raw)

    # ── Deduplicate & add date ──
    df = df.unique(subset=_key_cols)
    df = df.with_columns(pl.col("timestamp").dt.date().alias("date"))

    # ── Sentence lengths ──
    sent_len_cache = f"{data_dir}/sentence_lengths.parquet"
    if not os.path.exists(sent_len_cache):
        raise FileNotFoundError(f"{sent_len_cache} not found. Run `python compute_sentence_lengths.py` first.")
    print("Loading cached sentence lengths...")
    sent_len_df = pl.read_parquet(sent_len_cache)
    df = df.join(sent_len_df, on=_key_cols, how="left")

    mean_sentence_len = df.filter(pl.col("language") == "English")["mean_sentence_len_words"].mean()
    print(f"Mean sentence length (English, words per sentence): {mean_sentence_len:.2f}")

    # ── Pre-filtering stats ──
    print("\nSTATS – PRE-FILTERING")
    print("# Total Conversations:", len(df))
    print("# Unique Conversations:", df['conversation_hash'].n_unique())
    print("# Unique IPs:", df['hashed_ip'].n_unique())

    conv_per_ip = df['hashed_ip'].value_counts()
    print("Avg. Conversations per IP:", round(conv_per_ip['count'].mean(), 2))

    num_users_with_num_convs = conv_per_ip['count'].value_counts(name='conv_count_count')
    print("Greatest # Convs with >= 10 IPs:", num_users_with_num_convs.filter(pl.col('conv_count_count') >= 10)['count'].max())

    ip_stats = df.group_by("hashed_ip").agg([
        pl.len().alias("conv_count"),
        pl.col("country").n_unique().alias("unique_country_count"),
        pl.col("language").n_unique().alias("unique_lang_count"),
        pl.col("state").n_unique().alias("unique_state_count"),
    ])

    print("Avg. Countries per IP:", round(ip_stats['unique_country_count'].mean(), 2))
    print(ip_stats['unique_country_count'].value_counts().sort('unique_country_count'))
    print("Avg. Languages per IP:", round(ip_stats['unique_lang_count'].mean(), 2))
    print(ip_stats['unique_lang_count'].value_counts().sort('unique_lang_count').head(10))
    print("Avg. States per IP:", round(ip_stats['unique_state_count'].mean(), 2))
    print(ip_stats['unique_state_count'].value_counts().sort('unique_state_count'))

    # ── Filter IPs ──
    keep_ip = set(
        ip_stats.filter(
            (pl.col("conv_count") <= 161) &
            (pl.col("unique_country_count") <= 3) &
            (pl.col("unique_lang_count") <= 3) &
            (pl.col("unique_state_count") <= 3)
        )["hashed_ip"].to_list()
    )
    filt_df = df.filter(pl.col('hashed_ip').is_in(keep_ip))

    # ── Post-filtering stats ──
    print("\nSTATS – POST-FILTERING")
    print("# Total Conversations:", len(filt_df))
    print("# Unique Conversations:", filt_df['conversation_hash'].n_unique())
    print("# Unique IPs:", filt_df['hashed_ip'].n_unique())

    # Drop columns only needed during loading/filtering
    filt_df = filt_df.drop('conversation_hash', 'timestamp', 'state', 'country')

    # ── Day indices ──
    filt_df = filt_df.with_columns(pl.col('date').rank('dense').over('hashed_ip').alias('day_index'))
    filt_df = filt_df.with_columns(pl.col('day_index').max().over('hashed_ip').alias('day_group'))

    # ── Time series aggregations ──
    conv_by_day_df = filt_df.group_by("date").len().rename({"len": "conv_count"}).sort('date')
    user_by_day_df = filt_df.group_by("date").agg(pl.col("hashed_ip").n_unique().alias("user_count")).sort('date')
    mssgs_by_day_df = filt_df.group_by("date").agg(
        pl.col("turn").mean().alias("turn_count_mean"),
        (pl.col("turn").std() / pl.len().sqrt()).alias("turn_count_se"),
    ).sort('date')

    # ── Convert completion ──
    filt_df = filt_df.with_columns(
        pl.when(pl.col("completed") == "CompletionStatus.COMPLETED").then(1).otherwise(0).alias("completed")
    )
    complete_day_df = filt_df.group_by("date").agg(
        pl.col("completed").mean().alias("completed_mean"),
        (pl.col("completed").std() / pl.len().sqrt()).alias("completed_se"),
    ).sort("date")

    completion_rate_overall = round(filt_df['completed'].mean() * 100, 2)
    print(f"Completion Rate Overall: {completion_rate_overall}")

    # ── Intent & domain dummies (added to filt_df directly — single DataFrame) ──
    filt_df = make_dummies(filt_df, 'user_intent', 'IntentLabel.')
    intent_cols = [c for c in filt_df['user_intent'].unique().sort().to_list() if c is not None and c != "None"]
    filt_df = make_dummies(filt_df, 'conversation_domain', 'DomainLabel.')
    domain_cols = [c for c in filt_df['conversation_domain'].unique().sort().to_list() if c is not None and c != "None"]

    intent_day_mean_df = filt_df.group_by("date").agg(
        [pl.col(c).mean().alias(f"{c}_mean") for c in intent_cols] +
        [(pl.col(c).std() / pl.len().sqrt()).alias(f"{c}_se") for c in intent_cols]
    ).sort("date")
    domain_day_mean_df = filt_df.group_by("date").agg(
        [pl.col(c).mean().alias(f"{c}_mean") for c in domain_cols] +
        [(pl.col(c).std() / pl.len().sqrt()).alias(f"{c}_se") for c in domain_cols]
    ).sort("date")

    # ── Repeat user fraction ──
    first_seen = filt_df.group_by("hashed_ip").agg(pl.col("date").min().alias("first_date"))
    repeat_df = filt_df.join(first_seen, on="hashed_ip").with_columns(
        (pl.col("date") > pl.col("first_date")).cast(pl.Int8).alias("is_repeat")
    )
    repeat_by_day_df = (
        repeat_df.group_by("date")
        .agg(
            pl.col("is_repeat").mean().alias("repeat_frac"),
            (pl.col("is_repeat").std() / pl.len().sqrt()).alias("repeat_se"),
        )
        .sort("date")
    )

    # ── Sentence length by day ──
    sent_len_by_day_df = (
        filt_df.filter(pl.col("language") == "English")
        .group_by("date")
        .agg(
            pl.col("mean_sentence_len_words").mean().alias("sent_len_mean"),
            (pl.col("mean_sentence_len_words").std() / pl.len().sqrt()).alias("sent_len_se"),
        )
        .sort("date")
    )

    # ── By-model aggregations ──
    conv_by_model_day_df = filt_df.group_by(['model', 'date']).len().rename({"len": "conv_count"}).sort('date')
    user_by_model_day_df = filt_df.group_by(['model', 'date']).agg(
        pl.col("hashed_ip").n_unique().alias("user_count")
    ).sort('date')
    complete_model_day_df = filt_df.group_by(['model', "date"]).agg(
        pl.col("completed").mean().alias("completed_mean"),
        (pl.col("completed").std() / pl.len().sqrt()).alias("completed_se"),
    ).sort("date")
    intent_model_day_df = filt_df.group_by(['model', 'date']).agg(
        [pl.col(c).mean().alias(f"{c}_mean") for c in intent_cols] +
        [(pl.col(c).std() / pl.len().sqrt()).alias(f"{c}_se") for c in intent_cols]
    ).sort("date")
    domain_model_day_df = filt_df.group_by(['model', 'date']).agg(
        [pl.col(c).mean().alias(f"{c}_mean") for c in domain_cols] +
        [(pl.col(c).std() / pl.len().sqrt()).alias(f"{c}_se") for c in domain_cols]
    ).sort("date")

    # ── Activity-group filtering ──
    # Full version
    filt_df_ag = filt_df.filter(pl.col("day_group") <= 45)
    ag_versions = [(f"{OUTPUT_DIR}/wildchat-full", filt_df_ag, "")]

    # Truncated version: recompute day_index/day_group on pre-cutoff data.
    # Skipped entirely when cfg.cutoff is None.
    if cfg.cutoff is not None:
        _trunc_base = filt_df.filter(pl.col("date") < cfg.cutoff).drop(['day_index', 'day_group'])
        _trunc_base = _trunc_base.with_columns(pl.col('date').rank('dense').over('hashed_ip').alias('day_index'))
        _trunc_base = _trunc_base.with_columns(pl.col('day_index').max().over('hashed_ip').alias('day_group'))
        filt_df_trunc_ag = _trunc_base.filter(pl.col("day_group") <= 45)
        ag_versions.append((f"{OUTPUT_DIR}/wildchat-trunc", filt_df_trunc_ag, "_trunc"))

    m = WildChatMetrics(
        intent_cols=intent_cols,
        domain_cols=domain_cols,
        conv_by_day_df=conv_by_day_df,
        user_by_day_df=user_by_day_df,
        mssgs_by_day_df=mssgs_by_day_df,
        complete_day_df=complete_day_df,
        repeat_by_day_df=repeat_by_day_df,
        sent_len_by_day_df=sent_len_by_day_df,
        intent_day_mean_df=intent_day_mean_df,
        domain_day_mean_df=domain_day_mean_df,
        conv_by_model_day_df=conv_by_model_day_df,
        user_by_model_day_df=user_by_model_day_df,
        complete_model_day_df=complete_model_day_df,
        intent_model_day_df=intent_model_day_df,
        domain_model_day_df=domain_model_day_df,
        filt_df_all=filt_df,
        ag_versions=ag_versions,
        mean_sentence_len=mean_sentence_len,
        completion_rate_overall=completion_rate_overall,
        dataset=cfg,
    )

    # Compute template metrics inline from source conversations
    ensure_template_metrics(m, cache_path=None)

    if cache_path:
        print(f"Caching metrics to {cache_path}...")
        with open(cache_path, "wb") as f:
            pickle.dump(m, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("Metrics loaded.")
    return m


def ensure_template_metrics(m: WildChatMetrics, source_data_dir=None, cache_path=None,
                           min_count=100, prefix_len=500, force=False):
    """Compute templated-conversation daily counts if not already present, then re-save cache.

    Reads conversations from the source WildChat parquets, fingerprints each by
    its first `prefix_len` lowercased characters, and groups. Any fingerprint with
    >= `min_count` conversations whose prefix is exactly `prefix_len` chars long
    is considered a template.

    Dataset paths come from `m.dataset`; `source_data_dir` and `cache_path` may be
    passed to override.
    """
    if m.template_by_day_df is not None and m.template_groups_df is not None and not force:
        print("Template metrics already present.")
        return m

    cfg = m.dataset if m.dataset is not None else DATASETS["wildchat-4.8m"]
    if source_data_dir is not None:
        cfg = replace(cfg, source_data_dir=source_data_dir)
    if cache_path is None:
        cache_path = cfg.cache_path

    print("Computing template metrics from source conversations (may take a minute)...")
    text_df = load_source_text(cfg)
    text_df = text_df.with_columns(
        pl.col("Text").str.to_lowercase().str.slice(0, prefix_len).alias("fingerprint"),
        pl.col("Text").str.slice(0, prefix_len).alias("original_prefix"),
        pl.col("timestamp").dt.date().alias("date"),
    )

    # Identify template fingerprints
    _agg_exprs = [
        pl.len().alias("count"),
        pl.col("original_prefix").first().alias("original_prefix"),
        pl.col("language").first().alias("language"),
    ]
    if cfg.cutoff is not None:
        _agg_exprs += [
            (pl.col("date") < cfg.cutoff).sum().alias("count_before"),
            (pl.col("date") >= cfg.cutoff).sum().alias("count_after"),
        ]
    template_groups_df = (
        text_df.group_by("fingerprint")
        .agg(_agg_exprs)
        .filter(
            (pl.col("count") >= min_count)
            & (pl.col("fingerprint").str.len_chars() >= prefix_len)
        )
        .sort("count", descending=True)
    )
    template_fps = template_groups_df.select("fingerprint")
    print(f"Found {len(template_fps)} template groups (>= {min_count} convos, prefix >= {prefix_len} chars)")

    # Daily counts: template_count and total_count -> pct
    template_daily = (
        text_df.join(template_fps, on="fingerprint", how="semi")
        .group_by("date").agg(pl.len().alias("template_count"))
    )
    total_daily = text_df.group_by("date").agg(pl.len().alias("total_count"))

    template_by_day_df = (
        total_daily.join(template_daily, on="date", how="left")
        .with_columns(
            pl.col("template_count").fill_null(0),
            (pl.col("template_count").fill_null(0) / pl.col("total_count") * 100).alias("template_pct"),
        )
        .sort("date")
    )

    total_template = template_by_day_df["template_count"].sum()
    total_convos = template_by_day_df["total_count"].sum()
    print(f"Total templated conversations: {total_template:,} / {total_convos:,} ({total_template/total_convos*100:.1f}%)")

    m.template_by_day_df = template_by_day_df
    m.template_groups_df = template_groups_df

    # Per-fingerprint daily counts (for sparklines)
    template_daily_by_fp_df = (
        text_df.join(template_fps, on="fingerprint", how="semi")
        .group_by(["fingerprint", "date"])
        .agg(pl.len().alias("count"))
        .sort(["fingerprint", "date"])
    )
    m.template_daily_by_fp_df = template_daily_by_fp_df

    if cache_path:
        print(f"Re-saving cache to {cache_path}...")
        with open(cache_path, "wb") as f:
            pickle.dump(m, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("Template metrics ready.")
    return m
