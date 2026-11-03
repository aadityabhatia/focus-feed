# Education Content Feed System - Specifications

See proposal.md for problem, intent and scope.

## Functional Requirements

### Input & Processing
- Accept one or more files or directories as positional arguments for each subcommand
- Files must be actual file paths (absolute or relative to current directory)
- For each directory provided, process all relevant files within that directory recursively
- Skip files with JSON where all applicable steps complete AND file modification time matches JSON mtime. Treat missing JSON as new.
- Warn on orphaned JSON and special characters in filenames. Preserve filenames.
- Supported types: PDF, HTML, MP4, supported audio inputs, and video containers requiring audio extraction (see Audio Format Support)
- Check file modification time on each process; store mtime in JSON; use to verify if file was modified after processing

### Metadata
- Generate `<filename>.json` alongside source file.
- Fields: `title:{filename,extracted,generated}`, `description`, `summary`, `content`, `filename`, `filetype`, `date` ms, `publication_date`, `mtime`, `duration`, `creation_time`, `size`, `hash`, step maps `transcribe/summarize/extract/generate_title`, `error`.
- filepath is NOT stored; it is inferred from the JSON filename location.
- Title precedence: generated > extracted > filename. Auto-prepend parent folder name if missing. Sanitize filename key.
- Remove control chars 0x00-0x1f except tab/newline/CR and 0x7f from extracted PDF/HTML.
- Compute SHA256 during first extraction or before feed generation (whichever comes first).
- Transcribed audio/video: description blank, summary = key takeaways, content = transcript
- PDF/HTML: description = extracted metadata, content = plaintext, summary null unless summarized

### Feed Eligibility
- Transcribed audio/video: transcribe.complete AND summarize.complete
- PDF/HTML: extract.complete AND summarize.complete
- Ineligible items excluded from feeds
- Nested feeds: each feed includes all content from its entire subtree (inner feeds are subsets of outer feeds)

### Audio Format Support
- Accept these 15 media filename extensions for remote transcription:
  - **Pass through unchanged:** `.mp3`, `.m4a`, `.wav`, `.webm`, `.aac`, `.flac`, `.ogg`, `.opus`, `.mp4`, `.mpeg`, and `.mpga`. The original file is submitted to the transcription endpoint without audio extraction.
  - **Extract audio first:** `.mkv`, `.mov`, `.avi`, and `.wmv`. Extract the first audio stream to a temporary transcription-compatible audio file, then submit that file. Remove the temporary file after the transcription request completes or fails; preserve the source.
- Smart extraction MUST avoid unnecessary conversion: retain the source audio codec using stream copy/remux when the output container supports it. If that is not supported, decode and convert to FLAC to prioritize audio quality. The linked PyAV/FFmpeg build must support the input and required operations.
- File size is not a concern for extraction decisions; prioritize audio quality over minimizing temporary file size. The configured remote endpoint may still enforce its own upload or duration limits.
- If a video has no audio stream, or its container/codec cannot be processed, transcription MUST fail with a clear error and MUST NOT be marked complete.
- Transcribe supported audio and video inputs to markdown via the remote OpenAI-compatible API only (no local transcription). TRANSCRIPTION_ENDPOINT required.
- Summarize audio/video transcripts via LLM, feeding the entire transcript.
- Use PyAV for media container and stream metadata where available. Metadata values may be absent, container-level rather than stream-specific, or reported in stream time-base units.
- Record remote API errors in JSON.
- Feed eligibility for transcribed media remains `transcribe.complete` AND `summarize.complete`.

## PyAV Capability Assessment (`av>=11.0`)

### Technical Findings
- PyAV is a Python binding around FFmpeg libraries, including libavformat and libavcodec. It does not have a separate, fixed catalog of media formats. File/container demuxing and codec decoding are determined by the FFmpeg libraries and build linked to the installed PyAV package.
- `av.open(path)` normally autodetects the container. `container.streams.audio`, `.video`, and `.get(...)` expose streams. `container.decode(stream)` decodes packets to frames; `container.demux(stream)` yields encoded packets.
- Common formats are normally supported by standard FFmpeg builds, subject to the actual wheel/build and file contents:
  - Audio containers/inputs: MP3, WAV, Ogg (including Vorbis/Opus streams), M4A/MP4 audio, and AAC elementary streams.
  - Video containers/inputs: MP4, Matroska/MKV, AVI, and MOV.
  - Example codecs commonly encountered include MP3, AAC, PCM variants, Vorbis, Opus, FLAC, H.264/AVC, HEVC/H.265, VP8/VP9, and MPEG-4 Part 2. This is representative, not a guarantee or exhaustive support matrix. A container being readable does not mean every codec inside it is decodable.
- The installed build is authoritative. Where necessary, inspect available formats/codecs through PyAV/FFmpeg and perform representative fixture-based tests in the target deployment environment. The installed package version alone cannot promise support for every optional codec or muxer.

### Audio Stream Extraction and Conversion
- PyAV extracts the first audio stream to a separate temporary file for `.mkv`, `.mov`, `.avi`, and `.wmv` inputs.
- Recommended default for extraction without altering audio: select the desired audio stream, create an output container and stream using the input stream as a template when the target container supports that codec, demux only that input stream, skip flush packets with `packet.dts is None`, reassign each packet to the output stream, then mux it. This is packet remuxing/stream copy; it avoids decode/re-encode quality loss and is generally the efficient extraction path.
- Stream copy cannot change the audio codec and is limited by the target container's support for the source codec. For example, an AAC stream can be placed in a compatible MP4/M4A-style container, while an incompatible source/target pairing requires transcoding.
- PyAV also supports conversion: decode packets to audio frames, optionally use `AudioResampler` to change sample format, channel layout, or sample rate, encode frames with an output audio codec, and mux the resulting packets. This is transcoding and can introduce lossy quality changes. It requires the linked FFmpeg build to provide the requested decoder, encoder, and muxer.
- Therefore, the FFmpeg command-line executable is not inherently required for extraction or supported conversions. PyAV uses FFmpeg libraries. Use the FFmpeg CLI as an optional operational alternative when a command is simpler, a required filter/operation is not exposed or convenient in PyAV, or an external media tool is already part of deployment. Do not equate installing PyAV with installing the `ffmpeg` executable.

### Metadata
- Container-level metadata includes `container.metadata`, `container.duration`, `container.bit_rate`, `container.size`, and `container.start_time` where the format supplies them. Container duration uses `av.time_base` units (microseconds) and is converted to seconds by scaling with `av.time_base`; stream duration is instead expressed in that stream's `time_base`.
- Stream timing includes `stream.duration`, `stream.start_time`, and `stream.time_base`. Stream duration is expressed in `time_base` units and should be converted to seconds as `float(stream.duration * stream.time_base)` when both are present.
- Audio codec context exposes sample rate (`sample_rate` / `rate`), sample format, frame size, and channel layout. Stream codec context also identifies codec information. Video stream/codec context exposes video codec and frame/format properties.
- Bitrate, duration, creation-time tags, and other metadata are not guaranteed. Some containers omit them, report only aggregate/container values, or have inconsistent/malformed tags. Treat absent values as unknown rather than synthesizing authoritative values. A measured bitrate would require reading/estimating data and is distinct from a container-reported bitrate.

### Limitations and Common Issues
- Supported formats vary with bundled or system-linked FFmpeg, platform, packaging choices, optional libraries, and the particular stream codec. Test the actual production wheel/build and representative files.
- Container extension alone is not sufficient to determine whether a file can be decoded. Corrupt, truncated, encrypted, unusual-profile, or malformed media can still fail.
- Multiple audio streams are possible. Select a stream by explicit product policy (for example first/best stream, language, or user selection); do not assume there is exactly one.
- Packet remuxing preserves the encoded stream, not a normalized PCM representation. Target-container compatibility, timestamps, stream mapping, and final flush handling need care.
- Decode/re-encode conversion costs more CPU and may alter quality, channel layout, sample rate, duration boundaries, or metadata. Encoder and muxer availability is build-dependent.
- Metadata is incomplete or expressed in different time bases; duration and bitrate should be nullable and validated against sample files.
- Exact container and codec compatibility remains dependent on the installed PyAV/FFmpeg build and the configured transcription endpoint.

### Requirements for Format Processing
- The accepted filename extensions and extraction behavior are defined in Audio Format Support. Actual container/codec compatibility remains dependent on the installed build.
- The application MUST report unsupported container/codec or missing-audio errors clearly and MUST preserve current feed eligibility rules unless explicitly changed.
- Metadata extraction MUST tolerate unavailable duration, bitrate, sample rate, and tags without treating missing values as processing success or failure by themselves.
- Extraction MUST use stream copy where compatible and FLAC conversion otherwise; quality takes precedence over intermediate file size.
- Acceptance tests MUST cover direct-submit audio formats, video extraction formats, absent audio, multiple streams, unsupported codecs, and temporary-file cleanup against the deployed PyAV build.

### Documents
- Extract metadata + content from PDF/HTML.
- Summarization via explicit `summarize` command (required for feed eligibility).
- Use `publication_date` for feed pubDate if present.

### Reflection
- Check for `reflection.md` in source folder. Prepend its content to summarization prompt.
- Prompts: audio/video `PROMPT_PODCAST_REFLECTION` fallback `PROMPT_PODCAST`; PDF/HTML `PROMPT_ARTICLE_REFLECTION` fallback `PROMPT_ARTICLE`.

### Feeds
- Generate `podcast.xml` and `articles.xml` per folder with `feed.json`.
- Items include title, description, link `BASE_URL+relative_filepath`, pubDate, guid from hash.
- Sort by folder then filename. Markdown→HTML with XSS protection.
- Conditional generation: only if feed.json exists and eligible files present.
- Nested feeds: each feed includes all content from its entire subtree (inner feeds are subsets of outer feeds).
- Root `index.html` lists all feeds. Root feeds require feed.json in data root.
- Ensure hash present before feed generation.

### CLI
Commands: `process`, `watch`, `transcribe`, `summarize`, `extract`, `title`, `feeds`, `validate`, `create-feed`, `cleanup`
Accept one or more files or directories as positional arguments.
Flags: `--config-file`, `--force`, `--dry-run` for transcribe/summarize/extract.
`--force` re-runs step regardless of completion status or mtime.

## Non-Functional
- Local processing by default for extraction and feed generation; transcription is remote-only
- Open-weight models supported via OpenAI-compatible endpoints for transcription and summarization
- Summaries = concise key takeaways, transcripts = markdown
- Watch mode debounced with periodic rescan fallback

## Constraints
- Remote-only transcription; no local whisper
- Feeds only where feed.json exists
- Directory structure preserved
- Special characters preserved with warnings only
- Unix epoch timestamps with millisecond precision
- Title dict with filename/extracted/generated
- Markdown library (>=3.5.0) required for markdown to HTML conversion with XSS protection
- Remote transcription: vLLM server requires `VLLM_MAX_AUDIO_DECODE_DURATION_S` configuration for longer audio files
- PyAV media capabilities depend on its linked FFmpeg build; package version `av>=11.0` alone is not a universal codec/container guarantee
- No requirement currently exists to ship or invoke an FFmpeg CLI executable. PyAV's FFmpeg library dependency and the standalone CLI are separate deployment considerations.

## Risks
- Transcription quality varies → use established models
- LLM summarization inconsistent → tested prompts, configurable
- File watching may miss events → periodic rescan fallback
- Special characters → escape output, warn
- Control chars → removed during extraction
- Claiming universal format coverage based on PyAV version alone can cause runtime failures across builds.
- Remote APIs may reject otherwise readable files due to upload size, format, codec, or duration constraints.
- Transcoding can be slower and lossy compared with stream-copy extraction; packet remuxing may fail when the target container cannot carry the source codec.
- Missing or unreliable metadata can produce incorrect duration or publication information unless fields remain optional and are validated.
- Broadening extensions without updating scanning, validation, feed type rules, documentation, and tests creates inconsistent user expectations.

## Open Questions
- Is the requested outcome limited to recording this capability assessment, or should a later change expand Focus Feed's accepted inputs beyond MP4/PDF/HTML?
- Does the remote transcription endpoint accept audio-only and video containers, or only specific upload formats and size/duration ranges?
- Video extraction selects the first audio stream. Product configuration for selecting alternate language/commentary tracks is not currently specified.
- Should media metadata include audio stream sample rate, channels, and bitrate in the JSON schema, or remain limited to existing `duration` and `creation_time` fields?

## PyAV Documentation References
- `av.open`: https://pyav.org/docs/stable/api/_globals.html
- Container APIs, duration/bitrate and demux/mux: https://pyav.org/docs/stable/api/container.html
- Stream selection, codec context and timing: https://pyav.org/docs/stable/api/stream.html
- Audio sample properties and resampling: https://pyav.org/docs/stable/api/audio.html
- Packet remuxing cookbook: https://pyav.org/docs/stable/cookbook/basics.html
- PyAV 11.0.0 package information: https://pypi.org/project/av/11.0.0/

## Metadata Example
```json
{
  "title":{"filename":"lecture","extracted":"Lecture 1","generated":"..."},
  "description":"",
  "summary":"Key takeaways...",
  "content":"Transcript or text...",
  "filename":"lecture.mp4",
  "filetype":"mp4",
  "date":1710000000.123,
  "publication_date":1710000000.0,
  "mtime":1710000000.0,
  "duration":45.5,
  "creation_time":"2024-03-10T10:00:00Z",
  "size":1234567,
  "hash":"sha256...",
  "transcribe":{"status":"complete"},
  "summarize":{"status":"complete"},
  "extract":null,
  "generate_title":{"status":"complete"},
  "error":""
}
```

## Environment
Required: BASE_URL, LLM_ENDPOINT, LLM_API_KEY, LLM_MODEL, PROMPT_ARTICLE, PROMPT_PODCAST, PROMPT_TITLE, TRANSCRIPTION_ENDPOINT, TRANSCRIPTION_API_KEY
Optional: PROMPT_ARTICLE_REFLECTION, PROMPT_PODCAST_REFLECTION, TRANSCRIPTION_REMOTE_MODEL

## FFmpeg and PyAV Capability Research

### Findings
- FFmpeg is a media toolkit, not a single codec or format. `ffmpeg` uses demuxers to identify containers and expose elementary streams, decoders to produce raw frames, optional filters to transform frames, encoders to create coded streams, and muxers to write output containers. Input container support and output support are distinct from codec support.
- MP4, MKV/Matroska, AVI, MOV, and WMV/ASF are common container families. MP3, WAV, OGG, M4A, and AAC refer to common audio file/container or codec forms, not interchangeable types. A filename extension alone does not establish that the specific stream codec can be decoded or encoded.
- FFmpeg can inspect and process a large set of media types, but exact support depends on the installed build, enabled components, input codec, and output muxer/encoder. `ffmpeg -formats`, `-demuxers`, `-muxers`, `-decoders`, and `-encoders` reveal local capability. Consequently do not publish an unconditional exhaustive supported-format list.
- `ffprobe` reads container and stream information and can emit JSON, including format and stream fields and metadata tags. Useful fields include format name, duration, bit rate, tags, stream codec name/type, sample rate, channel layout/count, stream duration, and per-stream tags where available. Probe results can omit fields, and duration or timestamps may be absent, estimated, or container-level rather than specific to the selected audio stream.
- Project inspection: `pyproject.toml` requires `av>=11.0`. PyAV is a Python binding around FFmpeg libraries and is used by remote transcription and `services/audio_extractor.py`; `src/focus_feed/services/extractors.py` invokes the `ffprobe` executable through `subprocess` and parses JSON. The `ffmpeg` CLI is not used for audio extraction.

### Extraction approaches and quality
- **Stream copy/remux:** Select an audio stream and copy its compressed packets into a compatible audio container without decoding/re-encoding. This avoids generational quality loss and is fast, but only works when the output container supports that codec. It does not convert codec, sample rate, or channels. Example: `ffmpeg -i input.mp4 -map 0:a:0 -vn -c:a copy output.m4a`.
- **Decode and encode/transcode:** Decode input audio, optionally resample/downmix/filter, encode to a selected output codec/container. Example lossless PCM normalization: `ffmpeg -i input.mkv -map 0:a:0 -vn -ac 1 -ar 16000 -c:a pcm_s16le output.wav`. This changes audio data and, for lossy encoders, loses quality; resampling/downmixing can also change the signal.
- **Common lossy output:** `ffmpeg -i input.mov -map 0:a:0 -vn -c:a libmp3lame -q:a 2 output.mp3`. Availability of MP3 encoding depends on whether the FFmpeg build has the LAME external library enabled.
- `-map 0:a:0` makes audio stream selection explicit. For all audio streams use `-map 0:a`; use `-vn` to exclude video. Inputs can contain multiple languages or commentary tracks; selecting stream zero is a default only, not a guarantee of the desired language. Use `ffprobe` first where stream selection matters.
- To inspect audio streams in JSON: `ffprobe -v error -select_streams a -show_streams -show_format -of json input.mp4`. For narrow output, use `-show_entries` with requested fields. Probe JSON is suitable for automated parsing.
- For an audio stream in a video file, MP4/MKV/AVI/MOV/WMV are readable when the local FFmpeg build supports the container and stream codec. For output, WAV/PCM, MP3, OGG/Vorbis or Opus, and M4A/AAC are typical possibilities, but each requires a compatible muxer and encoder. AAC stream-copy into M4A is possible when input codec/container permit it; conversion requires an available AAC encoder.

### Transcription format recommendation
- Prefer sending the original audio stream or its original audio-only representation if the remote transcription endpoint accepts it. This avoids an unnecessary lossy generation and avoids time-consuming intermediate files.
- Audio extracted for transcription uses compatible stream copy when possible and FLAC fallback otherwise, as specified above. WAV is not required as an intermediate format.
- There is no universally best output: accepted formats, maximum upload size/duration, supported sample rates, channel constraints, and any model-specific preferences are external endpoint requirements and remain unconfirmed. Verify with the configured provider before making the format a product requirement.
- Preserve source files. Any converted audio should be treated as temporary processing data unless retention is explicitly required. Do not silently overwrite the video or retain large derivatives without a storage/cleanup policy.

### PyAV versus direct FFmpeg
- PyAV can open containers, inspect streams, demux packets, decode audio frames, resample, encode, and mux output. It supports the in-process extraction pipeline when the required operations are available in the linked FFmpeg libraries and PyAV API.
- Direct `ffmpeg` is not required merely because PyAV uses FFmpeg. Use the CLI when its concise stream mapping and filter/transcode workflow is preferred, when a tested command is easier to maintain than manual packet/frame handling, or when operations exposed by the CLI are not conveniently available through the installed PyAV version.
- PyAV is used for in-process audio extraction and metadata handling, avoiding an extra `ffmpeg` CLI executable dependency. Keep existing `ffprobe` dependency behavior explicit: it is optional at runtime today and metadata extraction warns/falls back when missing. If portable probing is needed, PyAV can also provide much of this information, but field coverage and parsing behavior must be validated against representative files.
- Neither approach guarantees support for every input. Distribution of PyAV binary wheels or system FFmpeg binaries should be checked for codec build features and platform compatibility. The existing project requirement `av>=11.0` does not itself specify the FFmpeg runtime version or exact codec set.

### Licensing and legal considerations
- FFmpeg's upstream codebase is generally LGPL v2.1 or later, but optional GPL components can cause a build to be GPL; certain nonfree configurations can make resulting binaries non-redistributable. External libraries can add separate obligations or change the effective licensing situation.
- Calling a separately installed FFmpeg executable and linking to FFmpeg libraries via PyAV are technically different distribution/deployment arrangements; neither should be assumed to eliminate license obligations. If the project distributes binaries, wheels, containers, or FFmpeg/PyAV dependencies, review the actual bundled build configuration, dependency notices, source availability, and applicable LGPL/GPL terms.
- FFmpeg's legal page is explicit that it is not legal advice. Patent rights in particular codecs may also apply independently of copyright licensing and vary by jurisdiction and use. Seek legal review for commercial redistribution or patent-sensitive deployment.
- Do not claim that all FFmpeg builds or codecs are under one license. Record the FFmpeg/PyAV wheel/build provenance before making a compliance conclusion.

### Proposed requirements and acceptance criteria
- Audio input expansion and extraction are specified in Audio Format Support. Video extraction selects the first audio stream and fails clearly when no audio stream is present.
- Extraction output must remain compatible with the configured transcription endpoint. Endpoint upload and duration limits remain externally defined.
- Conversion must avoid re-encoding when stream-copy is compatible; otherwise use the specified FLAC fallback and validate output against the configured endpoint.
- Probe/extraction failures and missing metadata must be non-fatal where possible, explicitly represented or logged, and must not be reported as successful audio extraction.
- Acceptance tests must cover supported direct-submit audio, MKV/AVI/MOV/WMV extraction, no-audio and multiple-audio-stream inputs, unsupported codecs, and endpoint-compatible output validation.

### Open questions
- Which transcription API implementation and model are deployed, and what exact audio MIME types, codecs, sample rates, channel layouts, duration, and upload-size limits does it accept?
- Should `ffprobe` remain a system executable dependency, be replaced by PyAV metadata inspection, or have a fallback strategy? The present software is mixed: PyAV is required, while ffprobe is called optionally.
- Is the project or any deployment redistributing PyAV/FFmpeg binaries, and which concrete FFmpeg build configuration and third-party codecs are included?

### Risks and assumptions
- Container and codec support is assumed to be adequate for representative target files but depends on runtime build configuration. This must be validated against real course media before promising format coverage.
- FLAC fallback extraction can produce large intermediate files and extend processing time; remote endpoint format limits remain external constraints.
- Multiple audio tracks, commentary, silence, corruption, DRM, unsupported codecs, and inaccurate metadata can prevent extraction or yield the wrong track.
- Licensing conclusions cannot be drawn solely from `av>=11.0`; linked-library build details and distribution model are not captured by that dependency declaration.

### References
- FFmpeg command documentation: https://ffmpeg.org/ffmpeg.html
- FFmpeg formats and muxers/demuxers: https://ffmpeg.org/ffmpeg-formats.html
- FFmpeg codecs and available encoders/decoders: https://ffmpeg.org/ffmpeg-codecs.html
- ffprobe documentation: https://ffmpeg.org/ffprobe.html
- FFmpeg licensing and legal notes: https://www.ffmpeg.org/legal.html
- PyAV documentation: https://pyav.org/docs/stable/
