# To learn more about how to use Nix to configure your environment
# see: https://developers.google.com/idx/guides/customize-idx-env
{ pkgs, ... }: {
  # Which nixpkgs channel to use.
  channel = "unstable";
  # Use https://search.nixos.org/packages to find packages
  packages = [
    pkgs.ollama
    pkgs.python312
    pkgs.uv
    pkgs.git
    pkgs.gcc-unwrapped
    pkgs.stdenv.cc.cc
  ];
  # Sets environment variables in the workspace
  env = {
    # This helps Python find the C++ libraries
    LD_LIBRARY_PATH = "${pkgs.stdenv.cc.cc.lib}/lib";
  };
  idx = {
    extensions = [ "ms-python.debugpy" "ms-python.python"];
    previews = {
      enable = true;
      previews = {};
    };
    workspace = {
      onCreate = {
        clone-repo = "git clone https://github.com/EnriqueVilchezL/ai_workshop_2025_gen_ai_session.git";
        
        setup-poetry = ''
          cd ai_workshop_2025_gen_ai_session/rag_app && \
          uv venv && uv pip install --compile -r pyproject.toml 
        '';
      };
      onStart = {
        ollama-serve = "ollama serve";
      };
    };
  };
}
