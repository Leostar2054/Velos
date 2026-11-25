function pack_fpm_for_velos(N)
%PACK_FPM_FOR_VELOS  MATLAB alternative to PACK_REAL_SAMPLES_FOR_VELOS.bat.
% Keeps all frames but only the central N x N pixels (default 512) of every
% acquisition folder under "Real Samples" that has positions.csv, and saves
% compressed .mat files (about 15-25 MB each) to "_velos_packed". Originals are only read.
if nargin < 1, N = 512; end
here = fileparts(mfilename('fullpath'));
root = fullfile(here, 'Real Samples');
out  = fullfile(here, '_velos_packed');
if exist(out, 'dir') ~= 7, mkdir(out); end
d = dir(fullfile(root, '**', 'positions.csv'));
CHUNK = 45;
for k = 1:numel(d)
    folder = d(k).folder;
    name = strrep(strrep(strrep(folder(numel(root)+2:end), filesep, '__'), ' ', '_'), '\', '__');
    meta_file = fullfile(out, sprintf('%s_c%d_meta.mat', name, N));
    if exist(meta_file, 'file') == 2, fprintf('already packed: %s\n', name); continue; end
    txt = fileread(fullfile(folder, 'positions.csv'));
    lines = regexp(strrep(txt, char(13), ''), '\n', 'split');
    lines = lines(~cellfun('isempty', strtrim(lines)));
    hdr = strtrim(regexp(lines{1}, ',', 'split'));
    jf = find(strcmp(hdr, 'file'), 1);
    files = cell(numel(lines) - 1, 1);
    for m = 1:numel(files)
        p = regexp(lines{m + 1}, ',', 'split');
        files{m} = strtrim(p{jf});
    end
    ok = cellfun(@(f) exist(fullfile(folder, f), 'file') == 2, files);
    if ~all(ok), fprintf('SKIP %s: %d image(s) missing\n', name, sum(~ok)); continue; end
    first = imread(fullfile(folder, files{1}));
    if size(first, 3) > 1, fprintf('SKIP %s: colour frames\n', name); continue; end
    [H, W] = size(first);
    n = floor(min([N, H, W]) / 2) * 2;
    r0 = floor((H - n) / 2);  c0 = floor((W - n) / 2);
    frames_all = zeros(n, n, numel(files), 'uint16');
    for m = 1:numel(files)
        a = imread(fullfile(folder, files{m}));
        frames_all(:, :, m) = a(r0+1:r0+n, c0+1:c0+n);
        if mod(m, 25) == 0, fprintf('  %s: %d/%d\n', name, m, numel(files)); end
    end
    parts = ceil(numel(files) / CHUNK);
    for q = 1:parts
        frames = frames_all(:, :, (q-1)*CHUNK+1:min(q*CHUNK, numel(files)));  %#ok<NASGU>
        first_index = (q-1)*CHUNK;  %#ok<NASGU>
        save(fullfile(out, sprintf('%s_c%d_part%dof%d.mat', name, N, q, parts)), 'frames', 'first_index', '-v7');
    end
    positions_csv = txt;  %#ok<NASGU>
    run_info_json = '';
    if exist(fullfile(folder, 'run_info.json'), 'file') == 2, run_info_json = fileread(fullfile(folder, 'run_info.json')); end %#ok<NASGU>
    crop_row0 = r0; crop_col0 = c0; frame_height = H; frame_width = W; crop = n; n_frames = numel(files); %#ok<NASGU>
    centre_full = imread(fullfile(folder, files{min(113, numel(files))})); centre_file = files{min(113, numel(files))}; %#ok<NASGU>
    save(meta_file, 'files', 'positions_csv', 'run_info_json', 'crop_row0', 'crop_col0', 'frame_height', ...
        'frame_width', 'crop', 'n_frames', 'parts', 'centre_full', 'centre_file', '-v7');
    fprintf('  -> %s packed\n', name);
end
fprintf('Done. Please tell Claude that packing has finished.\n');
end
