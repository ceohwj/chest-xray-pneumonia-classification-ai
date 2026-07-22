# Dataset Provenance and Use

## Known source

The chest X-ray classification dataset used in this project was provided to the
project author by **OZ Coding School** as educational course material for a
medical-AI project. The relevant program is the OZ Coding School healthcare/AI
training program:

- <https://ozcodingschool.com/ozcoding/healthcarecamp/b>

This statement identifies the channel through which the project author received
the dataset. It does **not** claim that OZ Coding School created, owns, or is the
original publisher of the underlying medical images.

## Local package audited by this project

| Component | Rows | Labels available |
| --- | ---: | --- |
| Training metadata | 5,216 | Yes |
| Test metadata | 624 | No |
| Sample submission | 624 | Placeholder only |

The training label mapping supplied for this project is:

- `0`: `NORMAL`
- `1`: `PNEUMONIA`

The local package uses standardized filenames such as `train_0001.png` and
`test_0001.png`. Patient identifiers, acquisition metadata, and a reliable
mapping back to original upstream filenames were not supplied. Consequently,
patient-level leakage cannot be audited from the available metadata.

## Provenance and licensing limits

The following information has not been established from the course materials
currently available in this repository:

- the original medical-image publisher or institution;
- an upstream dataset URL, DOI, or immutable version identifier;
- patient consent, de-identification, or ethics documentation;
- the license governing the underlying images;
- permission to redistribute the images outside the course context.

No public dataset should be inferred solely from matching class names or image
counts. Until OZ Coding School or the upstream rights holder supplies written
provenance and redistribution terms, this repository treats the image dataset
as course-provided, restricted material.

## Repository policy

- Source X-ray images are not committed to Git.
- Generated images that reproduce source X-rays, including Grad-CAM contact
  sheets and the image-containing final PDF, remain local-only.
- Model checkpoints and raw predictions remain excluded from the repository.
- Tracked split CSVs contain standardized filenames and labels only; they do not
  contain pixel data or patient identifiers.
- A person reproducing the experiments must obtain the dataset through an
  authorized course or rights-holder channel and follow the terms supplied by
  that provider.

## Suggested citation for this portfolio

Until upstream provenance is confirmed, describe the data conservatively as:

> Course-provided chest X-ray classification dataset, supplied by OZ Coding
> School for educational use; original upstream provenance and redistribution
> license not confirmed.

Do not replace this statement with a public dataset citation unless the images
or provider documentation establish that relationship.
