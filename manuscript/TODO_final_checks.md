# Final Submission Checklist

## Citations & References
- [ ] Replace placeholder citations in refs.bib with verified entries
  - [ ] Verify Rubin-Delanchy 2015 full author list
  - [ ] Verify Owen 2012 full author list
  - [ ] Verify MacGillavry 2013 full author list
  - [ ] Verify Ward 2013 citation details (convex hull null model)
  - [ ] Verify Levet 2015 full author list (SR-Tesseler)
- [ ] Add any missing biological parameter references
- [ ] Verify all \citep references compile correctly

## Author Information
- [ ] Add author names and affiliations
- [ ] Add corresponding author email
- [ ] Add ORCID identifiers
- [ ] Write acknowledgments section
- [ ] Add funding information

## FPR Calibration
- [ ] Consider running 500-rep or 1000-rep FPR calibration for tighter CIs
- [ ] Current: 200 reps, CI width ~0.10 at alpha=0.05
- [ ] With 500 reps: CI width ~0.06
- [ ] With 1000 reps: CI width ~0.04

## Experiments
- [ ] Consider running non-quick mode for experiments B-H (currently quick mode with 3 reps)
- [ ] Full mode would give 20-100 replicates per condition
- [ ] Priority: B (sensitivity) benefits most from more replicates

## Figures
- [ ] Verify all figure references match filenames in manuscript.tex
- [ ] Consider adding Figure 1: method schematic / workflow diagram
- [ ] Check figure resolution (currently 300 DPI)
- [ ] Verify colorblind-safe palette renders correctly in print

## Manuscript Content
- [ ] Proofread abstract (target: 150-250 words)
- [ ] Verify all numbers match JSON result files
- [ ] Check equation numbering is consistent
- [ ] Review Discussion for balance (strengths vs limitations)
- [ ] Ensure "methodological exploration" framing is clear throughout

## Journal Selection
- [ ] Choose target journal:
  - Bioinformatics (methods paper, shorter format)
  - PLoS Computational Biology (open access, comprehensive)
  - Nature Methods (high impact, brief communication)
  - BMC Bioinformatics (open access, full methods)
- [ ] Format manuscript to journal style
- [ ] Check word/page limits
- [ ] Prepare cover letter with journal-specific details

## Code & Data Availability
- [ ] Add repository URL to manuscript (currently "[repository URL]")
- [ ] Add license file to repository (MIT or BSD recommended)
- [ ] Verify all seeds produce reproducible results
- [ ] Tag a release version matching the submitted manuscript

## Final Steps
- [ ] Run `python3 -m pytest smlm_clustering/tests/ -v` — expect all tests pass
- [ ] Regenerate figures: `python3 -m smlm_clustering.validation.comprehensive_study --figures-only --output-dir results/final --figures-dir figures`
- [ ] Compile manuscript.tex and supplementary.tex — verify PDF output
- [ ] Check supplementary tables match main text numbers
- [ ] Have co-authors review and approve
