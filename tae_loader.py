"""Load a tiny video autoencoder (taeh3) as a normal VAE.

The de-rope pass 2 starts at sigma ~0.92, so its init latent only has to carry layout and choreography. The full
H3 VAE round trip on the slowed-down clip (decode pass 1, encode 2-3x as many frames) is wasted precision there; the
tiny taeh3 autoencoder does the same job ~10-20x faster. The final decode still uses a full VAE.
"""
import comfy.sd
import comfy.utils
import folder_paths


class FastDeropeTAELoader:
    """Load a video TAE from models/vae_approx as a normal VAE (the stock VAELoader only picks up a file named exactly
    taeh3.safetensors there)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "tae_name": (folder_paths.get_filename_list("vae_approx"),
                         {"tooltip": "madebyollin taeh3 with encoder + decoder (taeh3.safetensors from madebyollin/taehv)"}),
        }}

    RETURN_TYPES = ("VAE",)
    FUNCTION = "load"
    CATEGORY = "FastDerope"
    DESCRIPTION = ("Loads the tiny H3 autoencoder as a VAE for VAEEncode / VAEDecode. Use it for the throwaway steps of "
                   "a de-rope (decode pass 1 for the time smear, encode the slowed clip); keep the full H3 VAE for the "
                   "final decode.")

    _cache = {}

    def load(self, tae_name):
        if tae_name not in self._cache:
            sd = comfy.utils.load_torch_file(folder_paths.get_full_path_or_raise("vae_approx", tae_name))
            if not any(k.startswith("encoder.") for k in sd):
                raise RuntimeError(
                    f"{tae_name} has no encoder: it is the decoder-only preview TAE (~10 MB). The de-rope needs the "
                    "encoder+decoder taeh3 (~23 MB) from madebyollin/taehv; pick that file here.")
            vae = comfy.sd.VAE(sd=sd)
            vae.throw_exception_if_invalid()
            type(self)._cache = {tae_name: vae}
        return (self._cache[tae_name],)
