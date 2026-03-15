from pathlib import Path

import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
REPORT_DIR = BASE_DIR / "Daniya_EXML_Project_Report"

st.set_page_config(page_title="Certified Attribution Medical Imaging", layout="wide")

st.markdown(
    """
    <style>
    .stApp, .main {
        background-color: #FFFFFF !important;
        color: #111111 !important;
    }
    [data-testid="stAppViewContainer"],
    [data-testid="stHeader"],
    [data-testid="stToolbar"] {
        background-color: #FFFFFF !important;
    }
    html, body, [class*="css"] {
        font-size: 20px !important;
        background-color: #FFFFFF !important;
        color: #111111 !important;
    }
    h1 {
        font-size: 48px !important;
        font-weight: 700 !important;
        color: #111111 !important;
    }
    h2 {
        font-size: 36px !important;
        font-weight: 700 !important;
        color: #111111 !important;
    }
    h3 {
        font-size: 28px !important;
        font-weight: 650 !important;
        color: #111111 !important;
    }
    p, li, label, .stMarkdown {
        font-size: 20px !important;
        line-height: 1.6 !important;
        color: #111111 !important;
    }
    [data-testid="stSidebar"],
    [data-testid="stSidebar"] > div,
    section[data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        color: #111111 !important;
    }
    [data-testid="stSidebar"] * {
        color: #111111 !important;
    }
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] div {
        color: #111111 !important;
    }
    /* Fix selectbox / dropdown backgrounds */
    div[data-baseweb="select"] > div,
    div[data-baseweb="select"] > div > div,
    div[data-baseweb="popover"] div,
    div[data-baseweb="menu"],
    div[data-baseweb="menu"] li,
    ul[data-baseweb="menu"] li,
    [data-testid="stSelectbox"] div,
    [data-testid="stSelectbox"] span,
    .stSelectbox > div > div,
    .stSelectbox > div > div > div {
        background-color: #FFFFFF !important;
        color: #111111 !important;
    }
    /* Dropdown option hover */
    div[data-baseweb="menu"] li:hover,
    ul[data-baseweb="menu"] li:hover {
        background-color: #F0F0F0 !important;
        color: #111111 !important;
    }
    .workflow-title {
        font-size: 24px !important;
        font-weight: 700 !important;
        margin: 8px 0 14px 0 !important;
        color: #111111 !important;
    }
    .workflow-step {
        border: 2px solid #111111;
        border-radius: 12px;
        padding: 16px 18px;
        margin: 0 auto;
        max-width: 900px;
        background: #F8F9FA;
        color: #111111 !important;
        font-size: 19px !important;
        line-height: 1.5 !important;
    }
    .workflow-arrow {
        text-align: center;
        font-size: 28px !important;
        font-weight: 700 !important;
        margin: 6px 0;
        color: #111111 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

DATASETS = {
    "Brain MRI": {
        "slug": "brain_mri",
        "exp2_slug": "mri",
        "sample": "brain sample.png",
        "confusion": "brain_mri_confusion_matrix.png",
        "cert_folder": "mri",
    },
    "ChestX-ray14": {
        "slug": "chestxray",
        "exp2_slug": "chestxray",
        "sample": "chestsample.png",
        "confusion": "chestxray_confusion_matrix.png",
        "cert_folder": "chest",
    },
    "APTOS Fundus": {
        "slug": "fundus",
        "exp2_slug": "fundus",
        "sample": "fundus.png",
        "confusion": "fundus_confusion_matrix.png",
        "cert_folder": "fundus",
    },
    "ISIC Dermoscopy": {
        "slug": "isic",
        "exp2_slug": "isic",
        "sample": "isic sample.png",
        "confusion": "confusion_matrix.png",
        "cert_folder": "isic",
    },
}


def show_image(path: Path, caption: str | None = None) -> None:
    if path.exists():
        st.image(str(path), caption=caption, use_column_width=True)
    else:
        st.warning(f"Image not found: {path}")


def show_gallery(paths: list[Path]) -> None:
    if not paths:
        st.info("No images available in this section.")
        return
    for image_path in paths:
        show_image(image_path, image_path.name)


def show_gallery_compact(paths: list[Path], width: int = 700) -> None:
    if not paths:
        st.info("No images available in this section.")
        return
    for image_path in paths:
        if image_path.exists():
            st.image(str(image_path), caption=image_path.name, width=width)
        else:
            st.warning(f"Image not found: {image_path}")


def show_gallery_side_by_side(paths: list[Path], columns: int = 2) -> None:
    if not paths:
        st.info("No images available in this section.")
        return

    cols = st.columns(columns)
    for idx, image_path in enumerate(paths):
        with cols[idx % columns]:
            if image_path.exists():
                st.image(str(image_path), caption=image_path.name, use_column_width=True)
            else:
                st.warning(f"Image not found: {image_path}")


def workflow_step(title: str, description: str) -> None:
    st.markdown(
        f"""
        <div class="workflow-step">
            <strong>{title}</strong><br>
            {description}
        </div>
        """,
        unsafe_allow_html=True,
    )


def workflow_arrow() -> None:
    st.markdown('<div class="workflow-arrow">↓</div>', unsafe_allow_html=True)


def render_workflow(title: str, steps: list[tuple[str, str]]) -> None:
    st.markdown(f'<div class="workflow-title">{title}</div>', unsafe_allow_html=True)
    for index, (step_title, step_description) in enumerate(steps):
        workflow_step(step_title, step_description)
        if index < len(steps) - 1:
            workflow_arrow()


def page_overview() -> None:
    st.header("Overview")
    st.markdown(
        r"""
        **Project Goal**
        - Certify pixel-level attribution stability under input perturbations.
        - Compare attribution methods on four medical modalities.
        - Evaluate localization with synthetic grid construction (DiFull setting).

        **Certification Pipeline (high-level)**
        - Compute attribution heatmap $h(x)$.
        - Apply top-$K$ sparsification ($K \in \{50, 25, 5\}$).
        - Run randomized smoothing with $\sigma=0.15$ and $n=100$.
        - Certify each pixel as 1 / 0 / abstain using $\tau=0.75$.

        **Project Structure**
        - `src/certify/smoothing.py`: randomized smoothing and certification radius computation.
        - `src/certify/eval/robustness.py`: certified/abstain robustness metrics and plots.
        - `src/certify/eval/faithfulness.py`: deletion-based confidence/faithfulness curves.
        - `src/certify/eval/localization.py`: CertifiedGridPG localization metric.
        - `src/xai/attribution_unified.py`: IG, GradCAM, RISE, Occlusion, LRP unified interface.
        - `src/models/grid_multihead.py`: Grid multi-head model and DiFull wrapper support.
        - `server_scripts/certify_grid_isic_server.py`: grid certification pipeline execution.
        """
    )

    render_workflow(
        "Certification algorithm workflow",
        [
            (
                "1. Input and prediction",
                "Pass the medical image through the trained classifier and select the predicted or target class for explanation.",
            ),
            (
                "2. Attribution generation",
                "Compute a heatmap with Integrated Gradients, Grad-CAM, RISE, Occlusion, or LRP using the unified attribution interface.",
            ),
            (
                "3. Top-K sparsification",
                r"Keep only the most salient pixels at $K \in \{50\%, 25\%, 5\%\}$ so certification is applied to the highest-ranked attribution values.",
            ),
            (
                "4. Randomized smoothing",
                r"Add Gaussian noise with $\sigma = 0.15$ over $n = 100$ samples and estimate how stable each selected attribution decision remains.",
            ),
            (
                "5. Pixel certification",
                r"Apply threshold $\tau = 0.75$ to certify each pixel as important, unimportant, or abstain when confidence is insufficient.",
            ),
            (
                "6. Evaluation",
                "Summarize certified robustness, compute faithfulness curves, and compare certified regions with visible image structure.",
            ),
        ],
    )

    render_workflow(
        "Experimental setting workflow",
        [
            (
                "1. Dataset preparation",
                "Prepare Brain MRI, ChestX-ray14, APTOS Fundus, and ISIC with common preprocessing and train/validation/test splits.",
            ),
            (
                "2. Backbone training",
                "Train multiple pretrained CNN backbones and record confusion matrices plus validation accuracy before any explanation analysis.",
            ),
            (
                "3. Attribution benchmarking",
                "Generate explanations for each method on correctly predicted images and compare the visual behavior of different saliency algorithms.",
            ),
            (
                "4. Certification runs",
                "Run the certification algorithm for each attribution method and sparsity level to obtain certified-important, certified-unimportant, and abstain regions.",
            ),
            (
                "5. Experiment 1 and 2",
                "Experiment 1 studies 5 images with multiple architectures, while Experiment 2 scales to 100 images per dataset with ResNet-18.",
            ),
            (
                "6. Controlled localization check",
                "Build DiFull 2x2 grid data from ISIC so the target cell is known and measure localization quality with CertifiedGridPG.",
            ),
        ],
    )

    st.subheader("Attribution Fragility Examples")
    show_image(REPORT_DIR / "attribution sample" / "neuralnetwrokfragile.png", "neuralnetwrokfragile.png")
    show_image(REPORT_DIR / "attribution sample" / "pathadversialcam.png", "pathadversialcam.png")


def page_datasets() -> None:
    st.header("Datasets")
    dataset_name = st.selectbox("Select dataset", list(DATASETS.keys()))
    meta = DATASETS[dataset_name]

    st.markdown(
        """
        **Dataset comparison table**

        | Dataset | Modality | Task | Classes | Common Input | Split | Why included |
        |---|---|---|---:|---:|---|---|
        | ChestX-ray14 | X-ray | Pneumonia classification | 2 | 224x224 | 80/10/10 | Tests coarse thoracic pathology evidence |
        | Brain MRI | MRI | Brain tumor / normal classification | 4 | 224x224 | 80/10/10 | Tests structurally localized neuro-imaging evidence |
        | APTOS Fundus | Retinal fundus | Diabetic retinopathy grading | 5 | 224x224 | 80/10/10 | Tests diffuse lesion patterns in retinal images |
        | ISIC Dermoscopy | Skin lesion | Lesion classification | Multi-class | 224x224 | 80/10/10 | Tests fine-grained texture and artifact sensitivity |
        """
    )

    st.markdown("**Selected dataset details**")

    dataset_details = {
        "Brain MRI": [
            ("Modality", "Magnetic resonance imaging"),
            ("Prediction task", "4-class brain tumor / normal classification"),
            ("Visual pattern", "Tumor mass, shape, boundary, and surrounding tissue structure"),
            ("Why useful here", "Useful for checking whether certified regions remain concentrated around localized pathology"),
        ],
        "ChestX-ray14": [
            ("Modality", "Chest X-ray"),
            ("Prediction task", "Binary pneumonia classification"),
            ("Visual pattern", "Opacity and lung-region abnormalities"),
            ("Why useful here", "Useful for checking whether explanations remain stable on broad low-contrast evidence"),
        ],
        "APTOS Fundus": [
            ("Modality", "Retinal fundus photography"),
            ("Prediction task", "5-grade diabetic retinopathy severity classification"),
            ("Visual pattern", "Small lesions, exudates, and vascular changes"),
            ("Why useful here", "Useful for evaluating attribution methods on sparse retinal features"),
        ],
        "ISIC Dermoscopy": [
            ("Modality", "Dermoscopy"),
            ("Prediction task", "Multi-class skin lesion classification"),
            ("Visual pattern", "Pigment structure, borders, texture, and artifacts"),
            ("Why useful here", "Provides high visual variability and supports the grid-based localization experiment"),
        ],
    }

    detail_rows = dataset_details[dataset_name]
    details_table = "| Property | Value |\n|---|---|\n" + "\n".join(
        f"| {label} | {value} |" for label, value in detail_rows
    )
    st.markdown(details_table)

    st.subheader(f"Sample image panel: {dataset_name}")
    show_image(REPORT_DIR / "data sample" / meta["sample"], meta["sample"])


def page_model_training() -> None:
    st.header("Model Training")
    st.markdown(
        r"""
        **Training Details**
        - Architectures: ResNet-18, ResNet-50, DenseNet-121, EfficientNet-B0/B1, MobileNetV2.
        - Initialization: ImageNet pretrained backbones.
        - Optimizer: Adam with learning rate $1\times10^{-4}$.
        - Batch size: 32; early stopping with patience = 5.
        - Augmentation: rotation, horizontal flip, color jitter (RGB modalities).

        **Why ResNet-18 for large-scale certification**
        - Lower computational cost for repeated smoothing runs.
        - Stable baseline for cross-dataset comparison.
        - Used in Experiment 2 (100 images/dataset).

        **Validation Accuracy Table**
        Validation accuracy table from the report:

        | Dataset | ResNet-18 | ResNet-50 | DenseNet-121 | EfficientNet-B0 | EfficientNet-B1 | MobileNetV2 |
        |---|---:|---:|---:|---:|---:|---:|
        | ChestX-ray14 | 88.4 | 89.1 | 89.8 | 89.4 | 91.0 | 87.6 |
        | Brain MRI | 90.9 | 91.7 | 92.3 | 92.0 | 92.7 | 90.1 |
        | APTOS Fundus | 86.8 | 87.6 | 88.1 | 87.9 | 88.7 | 86.2 |
        | ISIC Dermoscopy | 84.4 | 85.5 | 86.2 | 85.9 | 86.9 | 83.8 |
        """
    )

    st.subheader("Confusion matrices")
    cms = sorted((REPORT_DIR / "confusion matrix").glob("*.png"))
    show_gallery(cms)


def page_attribution_methods() -> None:
    st.header("Attribution Methods")
    st.markdown(
        """
        **Attribution Methods (Implementation and Behavior)**
        - **Integrated Gradients:** integrates gradients along a straight path from baseline to input. In this project, the gradient path is sensitive to input perturbations, so the top-K set often changes across noisy samples and certification frequently abstains.
        - **Grad-CAM:** produces a coarse activation map from the last convolutional layer. It often certifies more pixels than the perturbation-based methods, but it can still return all certified-0 when activations are weak or spatially diffuse.
        - **RISE:** averages over many random binary masks. Because the estimate itself has high variance, the top-K set can change substantially across noisy samples.
        - **Occlusion:** measures score change under patch removal. Results depend on patch size and stride, and the resulting maps may be sparse after smoothing.
        - **LRP:** propagates relevance backward through the network. In these experiments it is often more structured than raw gradients, but certification still depends on how concentrated the relevance map is.

        **Why some methods produce all-certified-0 or abstain**
        - If the class activation map is very diffuse (e.g. background texture dominates), no pixel consistently passes top-K across all noisy samples.
        - If the activation peak is small relative to image size, at K=25% or K=5% that region may fall outside the sparsified mask entirely.
        - Dataset characteristics also matter: diffuse texture and low-contrast structure generally make certification harder.

        **Stability ranking (typical)**
        | Method | Certification tendency | Failure mode |
        |---|---|---|
        | GradCAM | Often certifies, but can be all-0 | Diffuse or weak activation blobs |
        | LRP | Moderate certification | Sharp maps hurt by input noise |
        | Occlusion | Low-moderate | Alignment and patch-size sensitivity |
        | Integrated Gradients | Rarely certifies | Gradient path instability under noise |
        | RISE | Almost never certifies | Intrinsic mask-sampling variance |

        **Interpretation focus**
        - Higher certified percentage means stronger stability under noise.
        - All-certified-0 indicates that the method does not provide enough evidence to certify any pixel as consistently important at the selected noise level.
        - Faithfulness is checked through confidence drop after deleting certified pixels.
        """

    )

    dataset_name = st.selectbox("Select dataset", list(DATASETS.keys()), key="attr_dataset")
    prefix_map = {
        "Brain MRI": "brain",
        "ChestX-ray14": "chest",
        "APTOS Fundus": "aptos",
        "ISIC Dermoscopy": "skin",
    }
    prefix = prefix_map[dataset_name]

    st.subheader(f"Attribution samples: {dataset_name}")
    paths = sorted((REPORT_DIR / "attribution sample").glob(f"{prefix}*.png"))
    show_gallery(paths)


def page_certification() -> None:
    st.header("Certification Results")
    dataset_name = st.selectbox("Select dataset", list(DATASETS.keys()), key="cert_dataset")
    meta = DATASETS[dataset_name]

    st.subheader(f"Certification panels: {dataset_name}")
    cert_dir = REPORT_DIR / "certifcation" / meta["cert_folder"]
    panels = sorted(cert_dir.glob("*.png")) if cert_dir.exists() else []
    show_gallery(panels)

    st.markdown(
        """
        **Result Interpretation (Certification Panels)**
        - Certified-1 pixels: pixels repeatedly selected as important under smoothing.
        - Certified-0 pixels: pixels repeatedly excluded by the top-K mask under smoothing.
        - Abstain: no confidence guarantee under current threshold.
        - Compare methods by certified coverage and abstention behavior.
        """
    )


def page_experiment_1() -> None:
    st.header("Experiment 1 (5 images, cross-architecture)")
    dataset_name = st.selectbox("Select dataset", list(DATASETS.keys()), key="exp1_dataset")
    slug = DATASETS[dataset_name]["slug"]

    st.markdown(
        """
        **Experiment 1 Design**
        - Small-scale controlled study: 5 images per dataset.
        - Multi-architecture comparison (cross-architecture behavior).
        - Purpose: examine method and architecture variability before larger-scale evaluation.

        """
    )

    render_workflow(
        "Experiment 1 workflow",
        [
            (
                "1. Pick 5 representative images",
                "Use a small, manageable evaluation subset per dataset so each architecture-method combination can be inspected carefully.",
            ),
            (
                "2. Run multiple backbones",
                "Evaluate explanations on ResNet, DenseNet, EfficientNet, and MobileNet families to test architecture sensitivity.",
            ),
            (
                "3. Certify each method",
                "Apply the same smoothing-based certification pipeline to IG, Grad-CAM, RISE, Occlusion, and LRP.",
            ),
            (
                "4. Compare robustness and faithfulness",
                "Read the robustness stack plots and deletion curves together to see whether stable maps are also informative.",
            ),
        ],
    )

    st.subheader("Robustness")
    show_image(
        REPORT_DIR / "experiment1" / "robustness" / slug / "robustness_stacked.png",
        "robustness_stacked.png",
    )

    st.subheader("Faithfulness")
    show_image(
        REPORT_DIR / "experiment1" / "faithfullness" / slug / "faithfulness_confidence_curves.png",
        "faithfulness_confidence_curves.png",
    )

    st.subheader("Experiment 1 overall images")
    overall = [
        REPORT_DIR / "experiment1" / "avg_robustness.png",
        REPORT_DIR / "experiment1" / "faithfullness" / "overall_confidence_curves.png",
    ]
    show_gallery([p for p in overall if p.exists()])


def page_experiment_2() -> None:
    st.header("Experiment 2 (100 images, ResNet-18)")
    dataset_name = st.selectbox("Select dataset", list(DATASETS.keys()), key="exp2_dataset")
    exp2_slug = DATASETS[dataset_name]["exp2_slug"]

    st.markdown(
        """
        **Experiment 2 Design**
        - Large-scale validation: 100 images per dataset.
        - Single model focus: ResNet-18 for computationally feasible smoothing runs.
        - Purpose: estimate method behaviour more reliably across modalities.
        """
    )

    render_workflow(
        "Experiment 2 workflow",
        [
            (
                "1. Fix one backbone",
                "Use ResNet-18 as the common classifier so the comparison focuses on attribution methods rather than architecture changes.",
            ),
            (
                "2. Scale to 100 images",
                "Increase the evaluation set for each dataset to reduce variance and obtain more stable aggregate statistics.",
            ),
            (
                "3. Repeat certification pipeline",
                "Generate attributions, sparsify them, run smoothing, and compute pixel-level certified labels for every method.",
            ),
            (
                "4. Aggregate by dataset and method",
                "Summarize percent certified and deletion-based faithfulness curves to identify robust trends across modalities.",
            ),
        ],
    )

    st.subheader("Robustness")
    show_image(
        REPORT_DIR / "experiment2" / "robustness" / exp2_slug / "robustness_stacked.png",
        "robustness_stacked.png",
    )

    st.subheader("Faithfulness")
    show_image(
        REPORT_DIR / "experiment2" / "faithfullness" / exp2_slug / "faithfulness_confidence_curves.png",
        "faithfulness_confidence_curves.png",
    )

    st.subheader("Experiment 2 overall images")
    overall = [
        REPORT_DIR / "experiment2" / "robustness" / "summary_mean_pct_certified.png",
        REPORT_DIR / "experiment2" / "robustness" / "method_by_dataset.png",
        REPORT_DIR / "experiment2" / "faithfullness" / "overall_confidence_curves.png",
    ]
    show_gallery([p for p in overall if p.exists()])


def page_grid_validation() -> None:
    st.header("Grid Validation")
    st.markdown(
        """
        **Grid Construction Pipeline**
        - Build synthetic 2x2 grids from ISIC images.
        - Each cell holds a class-specific lesion crop.
        - Ground-truth target region is known by construction.

        **DiFull Architecture Idea**
        - Training may use cell-wise logic, but DiFull attribution wrapper passes the full grid through backbone.
        - Target head output is used for attribution while gradients can flow across full image.
        - This makes it possible to inspect both target-cell evidence and context outside the target cell.
        """
    )

    render_workflow(
        "Grid / DiFull workflow",
        [
            (
                "1. Build 2x2 composite grids",
                "Combine four ISIC image cells into one synthetic image so the correct target location is known exactly.",
            ),
            (
                "2. Predict with DiFull model",
                "Use the grid-aware classifier to make the target-cell decision while still processing the full composite image.",
            ),
            (
                "3. Generate full-image attributions",
                "Explain the target output over the entire grid to reveal both correct evidence and possible contextual leakage.",
            ),
            (
                "4. Certify salient pixels",
                "Run the same smoothing-based attribution certification procedure on the grid explanations.",
            ),
            (
                "5. Measure CertifiedGridPG",
                "Check how much certified evidence falls inside the known target cell compared with the non-target cells.",
            ),
        ],
    )

    st.subheader("Grid construction samples")
    grid_samples = sorted((REPORT_DIR / "data sample").glob("grids*.png"))
    show_gallery_side_by_side(grid_samples, columns=2)

    st.subheader("Grid certification samples")
    grid_certs = sorted((REPORT_DIR / "certifcation").glob("grid_cert*.png"))
    show_gallery(grid_certs)

    st.markdown(
        """
        **Localization Result Interpretation**
        - CertifiedGridPG near 1.0: certified pixels concentrated in correct target cell.
        - Around the random baseline (0.25 for 2x2): limited localization.
        - Below the random baseline: certified pixels concentrate outside the target region.
        """
    )


def main() -> None:
    st.title("Pixel-Level Certified Explanations for Medical Imaging")

    page = st.sidebar.radio(
        "Sections",
        [
            "Overview",
            "Datasets",
            "Model Training",
            "Attribution Methods",
            "Certification Results",
            "Experiment 1",
            "Experiment 2",
            "Grid Validation",
        ],
    )

    if page == "Overview":
        page_overview()
    elif page == "Datasets":
        page_datasets()
    elif page == "Model Training":
        page_model_training()
    elif page == "Attribution Methods":
        page_attribution_methods()
    elif page == "Certification Results":
        page_certification()
    elif page == "Experiment 1":
        page_experiment_1()
    elif page == "Experiment 2":
        page_experiment_2()
    elif page == "Grid Validation":
        page_grid_validation()


if __name__ == "__main__":
    main()
