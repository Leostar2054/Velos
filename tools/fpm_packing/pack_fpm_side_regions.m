function pack_fpm_side_regions(N)
%PACK_FPM_SIDE_REGIONS  Second packing step for the Velos M4 evaluation.
% For ONE blood and ONE leaf acquisition, keeps all frames but only two N x N
% regions LEFT and RIGHT of the centre (the centre region is already packed).
% These are parts of the sample the AI model never sees during training.
% Output: "_velos_packed\<name>_L512_*.mat" and "_R512_*.mat". Originals are only read.
if nargin < 1, N = 512; end
here = fileparts(mfilename('fullpath'));
root = fullfile(here, 'Real Samples');
out  = fullfile(here, '_velos_packed');
if exist(out, 'dir') ~= 7, mkdir(out); end
sets = {fullfile(root, 'Blood', 'fpm_cs2100m_raster_20261123_222619'), ...
        fullfile(root, 'LEAF',  'fpm_cs2100m_raster_20261123_214040')};
tags = {'Blood__fpm_cs2100m_raster_20261123_222619', 'LEAF__fpm_cs2100m_raster_20261123_214040'};
CHUNK = 45;
for k = 1:numel(sets)
    folder = sets{k};
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
    first = imread(fullfile(folder, files{1}));
    [H, W] = size(first);
    r0 = floor((H - N) / 2);
    side = {'L', 'R'};  c0s = [64, W - 64 - N];
    stacks = {zeros(N, N, numel(files), 'uint16'), zeros(N, N, numel(files), 'uint16')};
    for m = 1:numel(files)
        a = imread(fullfile(folder, files{m}));
        for s = 1:2
            stacks{s}(:, :, m) = a(r0+1:r0+N, c0s(s)+1:c0s(s)+N);
        end
        if mod(m, 25) == 0, fprintf('  %s: %d/%d\n', tags{k}, m, numel(files)); end
    end
    for s = 1:2
        name = sprintf('%s_%s%d', tags{k}, side{s}, N);
        parts = ceil(numel(files) / CHUNK);
        for q = 1:parts
            frames = stacks{s}(:, :, (q-1)*CHUNK+1:min(q*CHUNK, numel(files))); %#ok<NASGU>
            first_index = (q-1)*CHUNK; %#ok<NASGU>
            save(fullfile(out, sprintf('%s_part%dof%d.mat', name, q, parts)), 'frames', 'first_index', '-v7');
        end
        positions_csv = txt; run_info_json = fileread(fullfile(folder, 'run_info.json')); %#ok<NASGU>
        crop_row0 = r0; crop_col0 = c0s(s); frame_height = H; frame_width = W; crop = N; n_frames = numel(files); %#ok<NASGU>
        centre_full = zeros(0, 0, 'uint16'); centre_file = ''; %#ok<NASGU>
        save(fullfile(out, sprintf('%s_meta.mat', name)), 'files', 'positions_csv', 'run_info_json', 'crop_row0', ...
            'crop_col0', 'frame_height', 'frame_width', 'crop', 'n_frames', 'parts', 'centre_full', 'centre_file', '-v7');
        fprintf('  -> %s packed\n', name);
    end
end
fprintf('Done. Please tell Claude that the side regions are packed.\n');
end
