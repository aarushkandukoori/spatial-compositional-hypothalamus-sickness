# Spatially Smoothed Compositional Models for Contrasting Hypothalamic Cell-Type Organization Across Sickness States: An Early-Stage Proposal

**Aarush Kandukoori** (Carnegie Mellon University) — Lead Student Author  
**Aditya Kandukoori** (Poolesville High School) — Co-Author

> **Status: early-stage research proposal.** No experimental results
> are claimed. We are actively seeking expert feedback.

## Abstract

Spatial transcriptomics such as MERFISH now lets labs map cell types across the hypothalamus under different physiological conditions, including induced sickness states. A common analysis question is where, and in which cell types, a condition shifts cell-type composition relative to healthy controls. Per-region tests ignore the spatial continuity of tissue and the nesting of tissue sections within animals. We outline an early-stage framework that models per-region cell-type counts with a multinomial logistic-normal model, places an intrinsic conditional autoregressive (ICAR) smoothness prior on condition effects across neighboring regions, and includes animal-level random effects. For several conditions, it also decomposes effects into a shared component and a condition-specific component, to separate a generic sickness signature from a pathogen-specific one. The planned evaluation uses only clearly labeled synthetic data with planted effects, and compares against per-region tests and non-spatial compositional baselines. No results are reported here; we are seeking expert feedback on whether the problem formulation and the assumptions are biologically sensible. AI tools (Claude) were used for literature synthesis, ideation, and draft preparation, an AI review checked the draft before release, and the authors take responsibility for all content.

## Introduction & Motivation

**Authors**

- Aarush Kandukoori, Carnegie Mellon University (Lead Student Author)
- Aditya Kandukoori, Poolesville High School (Co-Author)

**Status.** This is an early-stage proposal that seeks expert feedback. It contains no experimental results. All code planned for the accompanying repository uses synthetic, clearly labeled mock data only.

**Motivation.** Multiplexed FISH methods such as MERFISH [chen2015] have made it possible to profile the spatial location and transcriptional identity of many thousands of cells in the hypothalamus [moffitt2018]. Recent work from the Dulac lab describes MERFISH data from the murine hypothalamus in health and under induced sickness states, and a preprint on pathogen-specific sickness states. A natural downstream question is: *in which spatial regions and which cell types does a given condition shift composition, and which shifts are shared across conditions versus specific to one?*

Two features of such data make the question statistically delicate:

1. **Spatial continuity.** Neighboring regions of tissue are anatomically similar, so true condition effects are likely to vary smoothly or in contiguous patches. Independent per-region tests discard this structure and lose power in sparse regions.
2. **Replication structure.** Sections are nested within animals, and animals (not cells) are the unit of biological replication. Treating cells as independent produces overconfident results.

We propose a small, checkable modeling framework that addresses both, and we ask domain experts whether the formulation matches how such data are actually collected and interpreted.

## Related Work

**Spatial profiling of the hypothalamus.** MERFISH [chen2015] was used to molecularly and spatially profile the hypothalamic preoptic region and relate cell types to behavior [moffitt2018]. Our proposal is a statistical layer that could sit downstream of such cell-type annotation.

**Compositional differential abundance.** scCODA [buttner2021] models cell-type counts with a Bayesian Dirichlet-multinomial model and a logistic-normal-style parameterization with reference-type handling, but without spatial structure. Milo [dann2022] tests differential abundance on k-nearest-neighbor graph neighborhoods of cells in expression space, not in physical space.

**Spatial statistics for transcriptomics.** SpatialDE [svensson2018] identifies spatially variable genes using Gaussian process models. Graph-based learning on spatial neighbor graphs commonly builds on graph neural network ideas such as [kipf2017].

**Tooling and generative models.** Scanpy [wolf2018] and scVI [lopez2018] are standard components of single-cell pipelines; we expect to rely on the former for data handling in the repository and treat the latter as an optional upstream annotation tool.

To our knowledge, combining ICAR-style smoothing of condition effects over a physical-space region graph with compositional cell-type counts and a shared-plus-specific decomposition across conditions has not been packaged for this setting, but we may be missing relevant work and welcome pointers.

## Proposed Framework

**Setup.** Partition each tissue section into $R$ spatial bins (for example a regular grid, or anatomical sub-regions if available). Let $K$ be the number of annotated cell types. For section $s$ from animal $m(s)$ with condition label $c(s) \in \{0, 1, \dots, C\}$ (0 = healthy control), let $y_{s,r,k}$ be the count of cells of type $k$ in bin $r$ and $N_{s,r} = \sum_k y_{s,r,k}$.

**Likelihood.**
$$ y_{s,r,\cdot} \mid N_{s,r} \sim \mathrm{Multinomial}(N_{s,r}, \pi_{s,r}), \qquad \pi_{s,r,k} = \frac{\exp(\eta_{s,r,k})}{\sum_{j} \exp(\eta_{s,r,j})}. $$
The softmax is invariant to adding a constant to all $\eta_{s,r,j}$, so we fix a reference type $K$ for identifiability: $\alpha_{r,K} = \beta^{(c)}_{r,K} = u_{m,K} = 0$ for all $r$, $c$, $m$, which gives $\eta_{s,r,K} = 0$. All $k$-indexed parameters below (including $\alpha$, $\beta$, $\mu$, $b$, $u$, $\sigma_k$) therefore run over $k = 1, \dots, K-1$.

**Linear predictor.** Decompose each condition effect as $\beta^{(c)}_{r,k} = \mu^{(c)}_k + b^{(c)}_{r,k}$, where $\mu^{(c)}_k$ is a global condition effect and $b^{(c)}_{\cdot,k}$ is a spatial deviation. Then, for $k = 1, \dots, K-1$,
$$ \eta_{s,r,k} = \alpha_{r,k} + \sum_{c=1}^{C} \mathbb{1}[c(s)=c]\, \big(\mu^{(c)}_k + b^{(c)}_{r,k}\big) + u_{m(s),k}, \qquad u_{m,k} \sim \mathcal{N}(0, \sigma_k^2). $$
Here $\alpha_{r,k}$ is the control composition (log-odds relative to type $K$) in bin $r$, $\mu^{(c)}_k + b^{(c)}_{r,k}$ is the condition effect in bin $r$, and $u_{m,k}$ is an animal-level random effect that accounts for replication structure. The global effects get a weakly informative prior, $\mu^{(c)}_k \sim \mathcal{N}(0, s^2)$.

**Spatial smoothness prior.** Let $G=(V,E)$ be the bin adjacency graph with $|V|=R$, and $L$ its graph Laplacian ($L = D - A$). For each condition $c$ and type $k$, we place the intrinsic CAR prior on the spatial deviation $b^{(c)}_{\cdot,k}$:
$$ p(b^{(c)}_{\cdot,k} \mid \tau) \propto \exp\Big(-\tfrac{\tau}{2} \sum_{(r,r') \in E} (b^{(c)}_{r,k} - b^{(c)}_{r',k})^2 \Big) = \exp\Big(-\tfrac{\tau}{2}\, (b^{(c)}_{\cdot,k})^{\top} L\, b^{(c)}_{\cdot,k} \Big). $$
The identity $\sum_{(r,r')\in E}(b_r - b_{r'})^2 = b^{\top} L b$ holds for any vector $b$ on an undirected graph. This prior is improper because $L\mathbf{1} = 0$ (constant shifts are unpenalized). We resolve this by imposing the sum-to-zero constraint $\sum_r b^{(c)}_{r,k} = 0$, so the constant level of the condition effect is carried entirely by $\mu^{(c)}_k$ (and the control baseline by $\alpha_{r,k}$). If the graph has several connected components, $L$ has one null direction per component, so the sum-to-zero constraint is applied within each connected component. With a single $\mu^{(c)}_k$, this forces the mean effect to be equal across components; component-specific global terms are a possible extension.

**Shared plus specific decomposition (planned extension).** To separate a generic sickness signature from condition-specific ones, write
$$ \beta^{(c)}_{r,k} = \gamma_{r,k} + \delta^{(c)}_{r,k}, \qquad \gamma_{r,k} = \mu^{\gamma}_k + g_{r,k}, \qquad \delta^{(c)}_{r,k} = \mu^{\delta,(c)}_k + d^{(c)}_{r,k}. $$
That is, $\gamma$ and each $\delta^{(c)}$ get the same treatment: a global mean term ($\mu^{\gamma}_k \sim \mathcal{N}(0, s_\gamma^2)$, $\mu^{\delta,(c)}_k \sim \mathcal{N}(0, s_\delta^2)$) plus a sum-to-zero ICAR deviation ($g_{\cdot,k}$ with precision $\tau_\gamma$, $d^{(c)}_{\cdot,k}$ with precision $\tau_\delta$, constraints per connected component). This matches the base model with $\mu^{(c)}_k = \mu^{\gamma}_k + \mu^{\delta,(c)}_k$ and $b^{(c)}_{r,k} = g_{r,k} + d^{(c)}_{r,k}$. We take $\tau_\delta > \tau_\gamma$ and $s_\delta < s_\gamma$ (stronger shrinkage on the condition-specific terms) so that effects are attributed to the shared component unless the data demand otherwise. Identification: the likelihood identifies only the totals $\mu^{(c)}_k = \mu^{\gamma}_k + \mu^{\delta,(c)}_k$ for each condition and the totals $b^{(c)}_{r,k} = g_{r,k} + d^{(c)}_{r,k}$; the individual global terms $\mu^{\gamma}_k$ and $\mu^{\delta,(c)}_k$ are not separately identified by the data, since shifting a constant between them leaves every $\beta^{(c)}_{r,k}$ unchanged. The same holds for $g$ versus $d^{(c)}$. Note that $\gamma$ and $\delta^{(c)}$ are therefore only identified up to the choice of shrinkage; we flag this as an open question below.

**Inference.** The log-likelihood gradient with respect to $\eta_{s,r,k}$ is $y_{s,r,k} - N_{s,r}\,\pi_{s,r,k}$, which follows directly from differentiating the multinomial log-likelihood through the softmax. This makes MAP estimation by gradient methods, and Hamiltonian Monte Carlo for full posteriors, straightforward to implement. Decisions about 'where is there an effect' will use posterior intervals for $\beta^{(c)}_{r,k}$ and a posterior probability of $|\beta^{(c)}_{r,k}| > \epsilon$ for a user-chosen effect-size threshold $\epsilon$, with false discovery control by thresholding posterior inclusion-type quantities, to be specified after pilot work on synthetic data.

**Overdispersion.** If multinomial counts prove too tight relative to realistic variability, we will replace the likelihood with a Dirichlet-multinomial or add a bin-level Gaussian random effect on $\eta$.

## Methodology

**Planned repository modules.** (1) a synthetic tissue generator; (2) the model and inference code; (3) baseline implementations; (4) evaluation scripts and plots.

**Synthetic data statement.** The accompanying repository will use **synthetic, clearly labeled mock data only**. No real MERFISH or animal data are included, and no biological conclusions can be drawn from the repository. Synthetic sections are generated on a grid of bins with $K$ cell types, a smooth baseline composition, and planted condition effects in contiguous patches of bins, with animal-level random effects and tunable count depth.

**Procedure.**

1. Generate synthetic datasets across a grid of settings (number of animals per condition, cells per bin, effect size, patch size, overdispersion level).
2. Fit the proposed model and baselines to each dataset.
3. Score recovery of planted effects and calibration under null datasets with no planted effect.
4. Report sensitivity to the smoothness precision $\tau$ (fixed versus given a hyperprior).

**AI Assistance Disclosure.** AI tools (Claude) were used for literature synthesis, ideation, and draft preparation. An AI review checked the draft before release. The authors take responsibility for all content, including any errors.

## Planned Evaluation & Baselines

This section describes a protocol only. No results have been obtained.

**Baselines (planned).**

- **Per-bin tests:** for each bin and type, a Fisher exact or chi-square test of condition versus control, with Benjamini-Hochberg correction.
- **Non-spatial compositional model:** a scCODA-style [buttner2021] Bayesian model fit on whole-section counts, and on each bin independently.
- **Graph-neighborhood differential abundance:** a Milo-style [dann2022] approach, adapted so that the neighborhood graph is the physical-space graph rather than an expression-space graph.
- **Ablations of our model:** no spatial prior, no animal random effect.

**Metrics (planned).**

- On synthetic data with planted effects: precision, recall, and area under the precision-recall curve for detecting bin-and-type pairs with true nonzero effect; error in the estimated $\beta$ (root mean squared error).
- On null synthetic data: false-positive rate at nominal levels, and coverage of posterior intervals.
- Replication-structure stress test: increasing the animal-level variance $\sigma_k^2$ while holding cell counts fixed, to check whether methods that ignore animals lose calibration.
- Specificity test for the shared-plus-specific decomposition: planted shared and planted condition-specific patches, scoring correct attribution.

**Protocol.** Multiple random seeds per setting, with seeds and configurations recorded. Any claim about relative performance will be reported only for the synthetic settings tested, and will not be extrapolated to real data.

## Limitations & Open Questions

- **Synthetic-only plan.** Synthetic data reflect our modeling assumptions, so good performance on them says little about real tissue. A real evaluation would require collaboration and access to data and annotations from domain experts.
- **Binning.** Grid bins may cut across anatomical boundaries. Would anatomical registration or cell-level (rather than bin-level) modeling be preferred?
- **Cell-type annotation uncertainty.** We treat cell-type labels as known; in practice annotation errors propagate into counts.
- **Smoothness assumption.** ICAR smoothing may blur sharp anatomical boundaries where effects genuinely change abruptly. Edge-aware or adaptive weights are a possible extension.
- **Identifiability of shared versus specific effects** depends on the shrinkage choice, as noted above.
- **Biological relevance.** We do not know which compositional contrasts are of greatest interest to experts, nor whether composition (rather than expression changes within a cell type) is the right axis for sickness states. We would value guidance on this.
- **Related work coverage.** Our literature review was assisted by AI and limited in scope; we may have missed prior methods.

## References

- [moffitt2018] Moffitt et al., *Molecular, spatial, and functional single-cell profiling of the hypothalamic preoptic region*, Science 2018. https://doi.org/10.1126/science.aau5324
- [chen2015] Chen et al., *Spatially resolved, highly multiplexed RNA profiling in single cells*, Science 2015. https://doi.org/10.1126/science.aaa6090
- [wolf2018] Wolf et al., *SCANPY: large-scale single-cell gene expression data analysis*, Genome Biology 2018. https://doi.org/10.1186/s13059-017-1382-0
- [lopez2018] Lopez et al., *Deep generative modeling for single-cell transcriptomics*, Nature Methods 2018. https://doi.org/10.1038/s41592-018-0229-2
- [dann2022] Dann et al., *Differential abundance testing on single-cell data using k-nearest neighbor graphs*, Nature Biotechnology 2022. https://doi.org/10.1038/s41587-021-01033-z
- [buttner2021] Büttner et al., *scCODA is a Bayesian model for compositional single-cell data analysis*, Nature Communications 2021. https://doi.org/10.1038/s41467-021-27150-6
- [svensson2018] Svensson et al., *SpatialDE: identification of spatially variable genes*, Nature Methods 2018. https://doi.org/10.1038/nmeth.4636
- [kipf2017] Kipf et al., *Semi-Supervised Classification with Graph Convolutional Networks*, ICLR 2017. https://arxiv.org/abs/1609.02907
