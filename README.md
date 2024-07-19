# impl-solve-for-variants
A metadata implementation where variants are encoded in a hash and solved for prior to package selection.

This scheme is closely related to [Conda's build variants](
https://docs.conda.io/projects/conda-build/en/latest/resources/variants.html), at 
least in terms of how variants get encoded into the filename.

This scheme requires changes [to the solver in existing installers, such as pip](https://discuss.python.org/t/selecting-variant-wheels-according-to-a-semi-static-specification/53446/99), because the hash that encodes variant info means that the filename itself is no longer sufficient to sort distributions.

The following is taken from a working design doc, and may be out-of-date with current discussion. Please feel free to revise.

## Abstract
This PEP proposes allowing new content in the system platform tags. This new content will be used to represent variants and uniquify wheel filenames, which will allow multiple variants to coexist in repositories and prevent installation of conflicting shared libraries that implement a common interface.

## Rationale
Avoid installing incompatible shared libraries with a common interface
OpenMP is a commonly used standard for multithreading applications, often used in numerical packages. OpenMP is a standard, not a library, and several implementations of this standard are available as libraries. These separate libraries do not interact well when loaded into the same process, and result in either crashes, or undefined incorrect numerical results. PEP 725 improves this situation by allowing the OpenMP library dependency to be explicitly expressed. However, two wheels built with different OpenMP linkage will currently have the same filename, so these two wheels would not be able to coexist on one index, and would also possibly cause confusion with build caches.

By treating the shared interface as the key of a variant, installers would be able to avoid providing incompatible builds. For example, given binaries:

```
abc-1.0.0-omp_gnu.whl
abc-1.0.0-omp_intel.whl
```

The variant dimension here is “omp”. Installing one of these wheels would “activate” the variant, which would require any subsequent installs to match that variant.

## Simplify variants of packages

The ability to distribute wheels relies on matching platform compatibility information to ensure that wheels will work on the target system. The existing platform tags cover several fundamental operating system types and core libraries. There are several ways that packages can vary that the current platform tags do not account for. Such packages require workarounds from authors, because otherwise filenames are the same, where content is not. 

## Some of the schemes currently in use:

### Encode the variant information in the PEP440 local version identifier (e.g. jaxlib)
jaxlib-0.4.28+cuda12.cudnn89-cp39-cp39-manylinux2014_x86_64.whl
This scheme can’t be used directly on PyPI. PyPI disallows local version identifiers. These files are generally hosted on an external server, and a separate package on PyPI maps to them using the extras mechanism:

Provides-Extra: cuda12_cudnn89
Requires-Dist: jaxlib ==0.4.28+cuda12.cudnn89 ; extra == 'cuda12_cudnn89'

To the user, these specifications looks like:

jax[cuda12]

Any package not on PyPI will require the user to specify –-index-url, --extra-index-url, or --find-links, depending on whether the remote host is a PEP 503-compliant repo, or just a file collection.

### Encode the variant information in the project name (e.g. cudf-cu11 and cudf-cu12)
These packages share the same import name, and clobber each other in unpredictable ways if both are installed simultaneously
These make the PyPI web UI harder to use for finding a given project
Requires dependencies to these project to include specific implementation
This requires dynamic metadata, either by setup.py or by changing pyproject.toml for each package build

### Encode the variant information into the repo, serving many different repos
Package names are similar between repos, but package contents differ
Facilitates notion of default implementation - the default implementation is the one on PyPI, which doesn’t require any --extra-index-url or similar parameters
Locally cached wheels may be confused for remote wheels from a different repo (different variant)
Pytorch is one organization that implements this strategy. Their default implementation is hosted on PyPI as the “torch” package. To support older CUDA releases, they have a separate external repo, https://download.pytorch.org/whl/cu118. This repo mirrors torch’s dependency packages from PyPI, which allows the use of --index-url instead of --external-index-url, which reduces the likelihood of dependency confusion.
## Goals
* Add new metadata options in extensible ways to allow expression of currently known variants
* Capture additional information that affects binary compatibility
This may obviate other existing tags that effectively serve as proxies for one or more aspects of binary compatibility. For example, the manylinux system tags are primarily proxies for glibc version, but they also encode compatibility requirements with several other system libraries. This proposal would allow all dependencies to be expressed explicitly. This could improve compatibility detection and reporting behavior in installers.
* Avoid confusion with cached wheels. If a wheel is cached, it should always be exactly the desired variant.
* Prevent loading of incompatible shared libraries from being installed into the same environment
## What to add
We propose extending the platform tag mechanism to be arbitrarily extensible. The purpose of these additions are twofold:

* Capture information that affects binary compatibility, so that installers can avoid providing broken environments
* Capture dimensions of variability, to avoid invalid cache hits and unexpected unstated requirements

The “arbitrary” part of this proposal is negotiable. If there is a set of valid tags that captures known variability, we can limit the scope of this proposal. Either way, it is immediately concerning how many attributes could be present in the filename before tools break because of filename length. Web addresses have finite length, depending on the browser, and long filenames are a perennial issue in many programs. It does not seem wise to require all attributes to be added to the filename. As a means of storing attributes that are not expressible in the filename, we propose copying/moving the attributes from the filename into an arbitrarily extensible location. The core metadata file is an intuitive choice for this, especially in light of PEP 658, which specifies that the core metadata file should be served alongside the wheel that it describes. Unfortunately, several tools currently assume that the metadata must agree between sdists and all wheel files present for a project. Because of this stipulation, the variant metadata must be kept in a separate location, and will likely need to be served alongside the core metadata file.

Where the core metadata may be accessed at /files/distribution-1.0-py3.none.any.whl.metadata, the proposed variant file would be accessible at /files/distribution-1.0-py3.none.any.whl.variants.

With variant metadata stored in the variant metadata file, we still haven’t differentiated filenames. We propose to append the truncated hash of the variant metadata file to the filename. This would give us a jax filename like:

Jaxlib-0.4.28-cp39-cp39-manylinux2014_x86_64_mhdeadbeef.whl

### “But hashes don’t mean anything!”
It is common for people to manually inspect filenames on PyPI file download pages and simple API index pages. The hashes are opaque and disruptive to this behavior. Thus, as a compromise, we recommend that build tools populate the filename with available tags up to some globally-agreeable safe limit.

jaxlib-0.4.28-cp39-cp39-manylinux2014_x86_64_cuda12_cudnn89
_mhdeadbeef.whl

These may be used by install tools to filter the file list. Partial tags/values are not permitted, lest they be misinterpreted by an installer using them to filter filenames. However, the ultimate source of truth for identifying the correct distribution should always be drawn from the core metadata file, and installers must implement interpretation of the core metadata file when the file filtering scheme yields more than one result.

For example, these two distributions differ only in their hash, and the resolver must access the variant metadata to decide which is preferred.

jaxlib-0.4.28-cp39-cp39-manylinux2014_x86_64_cuda12_cudnn89
_mhdeadbeef.whl
jaxlib-0.4.28-cp39-cp39-manylinux2014_x86_64_cuda12_cudnn89
_mh01234abc.whl
Optional hashes
Hashes serve two purposes:
Giving a place to dump metadata that may be helpful, but is not immediately interesting to humans
Differentiating files based on metadata that people deem not important enough to be part of the filename

If people really don’t like hashes for whatever reason, the hashes should not be mandatory. It is likely that human-readable tags in the filename will be enough for many use cases. If people omit hashes, the worst-case scenario is that their packages end up confused like they are today, but collisions would be lower because more metadata tags will be allowed in the filename.
New tools
The packaging project is responsible for doing hardware detection and translating the system state into tags that can be used by the installer resolver to filter filenames. This is then vendored into popular installer tools, such as UV and pip. There are two options to extend the platform tags:

* Extend the current packaging project to account for several new cases
* Augment packaging so that it can accept plugins

We propose the latter option, both because predicting future usage is difficult, and because the intention of this proposal is to not incur additional maintenance burden on existing maintainers. We’ll refer to these packages below as packaging-selector packages, and we expect that there would be several of them, unified by a standardized entry point or interface. For example, there may be a packaging-selector-x86_64 or packaging-selector-cpu or packaging-selector-nvidia. These packages would contain the logic for:


* Reading preferences from user configuration
* Detecting hardware metadata from the system state
* Formatting user preferences and hardware metadata into tags, return them to packaging for their inclusion in a packaging tag

The behavior applies in the same way both when producing a package and when consuming a package.

## Mechanics of building an sdist
In order to use a packaging-selector package, it must be specified in a project’s pyproject.toml. This could be either one entry for build time and one for runtime, or a single key that applies to both the [build-system] and the [project] tables. The packaging-selector package would be necessary to run the build tool, so it definitely belongs in the [build-system] table. The runtime metadata will also need to know which packaging-selector package to use to match the correct wheel.

Parameters passed to the build package with the --config-settings parameter override any configuration from pip.conf or other config file, with the intent that build systems should be able to easily override any user settings or static hardware configuration.

### Sorting
Tag sorting is an arbitrary thing, and consistency matters more than anything. For an initial proposal of sorting, we propose:

* Preserve the sort order of existing platform tags
* Allow the packaging-selector packages to define a per-tag weight
* Sort combined set of tags from selector packages according to weights
* Sort alphabetically to resolve anything with similar weight
Default package specification
With variation comes ambiguity. Installers must handle this ambiguity somehow. This proposal suggests that the package author may indicate a default variant value for each variant in their package. This package-provided default would be used as a last resort, if no user configuration selected a variant, and if no other packages in the environment have selected a variant.
Mechanics of selecting variants at install time
Pre-installation of packaging-selectors must be done before one can install a package that uses selectors. To make this work, we propose that the repository-level metadata include a list of any necessary packaging-selectors. The installer would match this against its known capabilities (either vendored code or the list of installed packages. If any package-selector capability was missing, the installer should exit and prompt the user to install the missing package-selector(s).

Assuming that all necessary packaging-selector capabilities are in place, the installer uses the packaging package as usual, and the packaging package iterates through the packaging-selector(s) to obtain the list of tags. These tags are then sorted and used to form the platform tag.

TODO: is this a good place for writing out different user scenarios?
Rejected Alternatives
Moving hardware-specific implementation to plugins
This is more immediately possible with existing standards (demonstrated by`pip install -U "jax[cuda12]"`), but it has higher development and maintenance costs for developers.
Adding metadata attributes directly to the platform tag (i.e. do not use the hash scheme)
We have observed issues with very long wheel filenames. We would like to avoid potential issues here. The hash is admittedly opaque, but we suggest a mitigation in the implementation section above.
Replace the existing platform tag with the new information
The current platform tags overlap with metadata that this proposal will produce. That implies wasted filename space from duplicated information. Removing the older platform tags that are less specific could free up some space for more specific tags.

jaxlib-0.4.28-cp39-cp39-x86_64_glibc217_cuda12_cudnn89
_mhdeadbeef.whl
We believe this change would be too disruptive in making older versions of pip and other installers unable to use newer wheels.


