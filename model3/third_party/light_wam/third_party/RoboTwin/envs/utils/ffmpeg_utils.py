import shutil


def get_ffmpeg_executable() -> str:
    try:
        import imageio_ffmpeg

        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        if ffmpeg_exe:
            return ffmpeg_exe
    except Exception:
        pass

    ffmpeg_exe = shutil.which("ffmpeg")
    if ffmpeg_exe:
        return ffmpeg_exe

    raise FileNotFoundError(
        "No ffmpeg executable found. Install `imageio-ffmpeg` in the active environment or "
        "make sure a system `ffmpeg` binary is available on PATH."
    )
