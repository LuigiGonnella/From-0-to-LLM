% Carica il file .mat
data = load('imagelabels.mat');

% Supponiamo che il file contenga una variabile chiamata 'labels'
labels = data.labels; 


ids = load('setid.mat');

trnid = ids.trnid; 

tstid = ids.tstid; 

