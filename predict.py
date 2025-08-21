import torch, base64, io, soundfile as sf, numpy as np
import nemo.collections.asr as nemo_asr
from huggingface_hub import hf_hub_download

class Predictor:
    def setup(self):
        model_path = hf_hub_download(
            repo_id="ai4bharat/indicconformer_stt_sa_hybrid_rnnt_large",
            filename="indicconformer_stt_sa_hybrid_rnnt_large.nemo",
            local_dir="/tmp/models",
            local_dir_use_symlinks=False
        )
        print("🔄 Loading NeMo model...")
        self.model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.restore_from(model_path)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(device)
        print(f"✅ Model loaded on {device}")

    def predict(self, wav_base64: str) -> dict:
        try:
            audio_bytes = base64.b64decode(wav_base64)
            audio_np, sr = sf.read(io.BytesIO(audio_bytes))
            if audio_np.ndim == 2:
                audio_np = audio_np[:, 0]  # mono
            result = self.model.transcribe(
                [np.ascontiguousarray(audio_np, dtype=np.float32)],
                language_id="sa"
            )
            if isinstance(result[0], list):
                final_result = result[0][0]
            elif isinstance(result[0], tuple):
                final_result = result[0][1]
            else:
                final_result = result[0]
            return {"transcript": final_result.strip()}
        except Exception as e:
            import traceback
            return {"error": str(e), "traceback": traceback.format_exc()}
