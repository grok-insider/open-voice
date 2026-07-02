{
  description = "open-voice — agnostic speech-to-text + text-to-speech (local Rust engines + OpenAI, ElevenLabs, Cartesia, xAI)";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  # Prebuilt closures are pushed to the grok-insider cachix cache by CI, so
  # consumers never compile open-voice.
  nixConfig = {
    extra-substituters = [
      "https://grok-insider.cachix.org"
      "https://nix-community.cachix.org"
    ];
    extra-trusted-public-keys = [
      "grok-insider.cachix.org-1:ZxLVOxJ1CjdY3vQl1I99qCtwNZwIU4+/QwqSvntB/5w="
      "nix-community.cachix.org-1:mB9FSh9qf2dCimDSUo8Zy7bkq5CX+/rkCWyvRCYg3Fs="
    ];
  };

  outputs = { self, nixpkgs }:
    let
      lib = nixpkgs.lib;
      systems = [ "x86_64-linux" "aarch64-linux" ];
      forAllSystems = lib.genAttrs systems;

      packageFor = system:
        let
          pkgs = import nixpkgs { inherit system; };
          version = (lib.importTOML ./Cargo.toml).workspace.package.version;
        in
        pkgs.rustPlatform.buildRustPackage {
          pname = "open-voice";
          inherit version;
          src = ./.;

          cargoLock.lockFile = ./Cargo.lock;

          # Default features only: the `local` ONNX feature is excluded here
          # because `ort-sys` downloads ONNX Runtime binaries at build time,
          # which the Nix sandbox forbids. Local inference users build with
          # `cargo build --features local` (see README).
          cargoBuildFlags = [ "-p" "ov-cli" ];
          cargoTestFlags = [ "--workspace" ];

          # TLS is rustls end-to-end; ring only needs the stdenv C compiler.
          nativeBuildInputs = [ pkgs.makeBinaryWrapper ];

          # ffmpeg powers decode/probe/transcode at runtime.
          postFixup = ''
            wrapProgram "$out/bin/openvoice" \
              --prefix PATH : "${lib.makeBinPath [ pkgs.ffmpeg ]}"
          '';

          meta = {
            description = "Agnostic speech-to-text + text-to-speech CLI (openvoice)";
            homepage = "https://github.com/grok-insider/open-voice";
            license = lib.licenses.mit;
            mainProgram = "openvoice";
            platforms = systems;
          };
        };
    in
    {
      packages = forAllSystems (system: rec {
        default = packageFor system;
        open-voice = default;
      });

      apps = forAllSystems (system: {
        default = {
          type = "app";
          program = "${self.packages.${system}.default}/bin/openvoice";
          meta.description = "open-voice CLI (openvoice)";
        };
      });

      # Home Manager module: installs `openvoice` (prebuilt from the cache),
      # optionally renders the config file and the tt-*/sp-* aliases.
      #
      # Secrets policy: API keys are NOT managed here — they must never enter
      # the Nix store. Export XAI_API_KEY / ELEVENLABS_API_KEY / ... in the
      # session environment instead.
      homeManagerModules.default = { config, lib, pkgs, ... }:
        let
          cfg = config.programs.open-voice;
        in
        {
          options.programs.open-voice = {
            enable = lib.mkEnableOption "open-voice (openvoice CLI)";

            package = lib.mkOption {
              type = lib.types.package;
              default = self.packages.${pkgs.stdenv.hostPlatform.system}.default;
              defaultText = lib.literalExpression "open-voice.packages.\${system}.default";
              description = "The open-voice package providing `openvoice`.";
            };

            settings = lib.mkOption {
              type = lib.types.nullOr (pkgs.formats.toml { }).type;
              default = null;
              example = {
                defaults = {
                  stt_provider = "auto";
                  language = "es";
                };
                providers.cartesia.tts_voice = "some-voice-id";
              };
              description = ''
                Declarative `~/.config/open-voice/config.toml`. When null the
                CLI runs on built-in defaults and the user manages the file by
                hand. Never put API keys here — use environment variables.
              '';
            };

            aliases.enable = lib.mkEnableOption
              "shell aliases (tt-en/tt-es/tt-ru for transcribe, sp-* for speak)";
          };

          config = lib.mkIf cfg.enable (lib.mkMerge [
            { home.packages = [ cfg.package ]; }

            (lib.mkIf (cfg.settings != null) {
              xdg.configFile."open-voice/config.toml".source =
                (pkgs.formats.toml { }).generate "open-voice-config.toml" cfg.settings;
            })

            (lib.mkIf cfg.aliases.enable {
              home.shellAliases = {
                tt-en = "openvoice transcribe --lang en";
                tt-es = "openvoice transcribe --lang es";
                tt-ru = "openvoice transcribe --lang ru";
                sp-en = "openvoice speak --lang en";
                sp-es = "openvoice speak --lang es";
                sp-ru = "openvoice speak --lang ru";
              };
            })
          ]);
        };

      checks = forAllSystems (system: {
        default = self.packages.${system}.default;
      });

      devShells = forAllSystems (system:
        let
          pkgs = import nixpkgs { inherit system; };
        in
        {
          default = pkgs.mkShell {
            name = "open-voice-dev";
            packages = with pkgs; [
              cargo
              rustc
              rustfmt
              clippy
              rust-analyzer
              ffmpeg
              # For `--features local`: ort-sys' build script downloads ONNX
              # Runtime via ureq/native-tls and needs these on the host.
              pkg-config
              openssl
            ];
          };
        });
    };
}
