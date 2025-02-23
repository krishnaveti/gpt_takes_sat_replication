# hash:sha256:d8b75f19f3b492b8dcad91ca12a98b3a7771e7b5883a9cdba5cf8c1e6b12f73b
FROM registry.codeocean.com/codeocean/julia:1.10.2-22.04.1

ARG DEBIAN_FRONTEND=noninteractive

RUN julia -e 'using Pkg; Pkg.add(PackageSpec(name="CSV", version="0.10.15"))' \
    && julia -e 'using Pkg; Pkg.add(PackageSpec(name="Distributions", version="0.25.116"))' \
    && julia -e 'using Pkg; Pkg.add(PackageSpec(name="Plots", version="1.40.9"))' \
    && julia -e 'using Pkg; Pkg.add(PackageSpec(name="StatsPlots", version="0.15.7"))' \
    && julia -e 'using Pkg; Pkg.add(PackageSpec(name="XLSX", version="0.10.4"))' \
    && julia -e 'using Pkg; Pkg.API.precompile()'

ADD "https://github.com/coder/code-server/releases/download/v4.95.3/code-server-4.95.3-linux-amd64.tar.gz" /.code-server/code-server.tar.gz
	
RUN cd /.code-server \
    && tar -xvf code-server.tar.gz \
    && rm code-server.tar.gz \
    && ln -s /.code-server/code-server-4.95.3-linux-amd64/bin/code-server  /usr/bin/code-server

RUN mkdir -p /.vscode/extensions \
    && code-server --extensions-dir="/.vscode/extensions" --install-extension REditorSupport.R \
    && code-server --extensions-dir="/.vscode/extensions" --install-extension continue.continue \
    && code-server --extensions-dir="/.vscode/extensions" --install-extension ms-python.python \
    && code-server --extensions-dir="/.vscode/extensions" --install-extension ms-toolsai.jupyter \
    && code-server --extensions-dir="/.vscode/extensions" --install-extension reageyao.bioSyntax \
    && code-server --extensions-dir="/.vscode/extensions" --install-extension saoudrizwan.claude-dev
