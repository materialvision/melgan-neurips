with import ./nix/nixpkgs.nix {

  overlays = [

    (self: super:
      {
        blas = super.blas.override {
          blasProvider = self.mkl;
        };
        lapack = super.lapack.override {
          lapackProvider = self.mkl;
        };
      }
    )
  ];

};

let
  py = python3;
in
mkShell {
  buildInputs = [

    ffmpeg
    # ffmpeg-full
    entr

    (py.withPackages (ps: with ps; [

      librosa
      # ffmpeg-full leads to assertion error with blas override.
      (pydub.override { ffmpeg-full = ffmpeg; })
      click
      pyyaml
      pytorchWithCuda
      scipy

      # To install packages not in nixpkgs
      pip

      # dev deps
      pudb  # debugger
      ipython
    ]))
   ];

  shellHook = ''
    export PIP_PREFIX="$(pwd)/.build/pip_packages"
    export PATH="$PIP_PREFIX/bin:$PATH"
    export PYTHONPATH="$PIP_PREFIX/${py.sitePackages}:$PYTHONPATH"
    unset SOURCE_DATE_EPOCH
  '';
}
