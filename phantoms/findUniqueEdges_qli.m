function [uniq_edges, uniq_faces] = findUniqueEdges_qli(V, C, idx_bounded, max_edge_per_face)

if nargin <4
    max_edge_per_face = 10
end

nbofCell = length(C);
nbofVertices = length(V);
Edges = []; Faces = [];
for i = 1:nbofCell
    indVert = C{i}; % index of vertices in Cell i
    if ~ismember(1, indVert)
        vertices = V(indVert,:); %here in vertices index 1:end maps to indVert(1:end)
        K = convhulln(vertices); %triangular faces from convhulln;
        [V2 F2] = mergeCoplanarFaces(vertices,K); %V2 same as vertices, F2 cell structured-each cell contains all edges
        for j = 1:length(F2)
            tmp = [F2{j}, F2{j}(1)];
            %if length(F2{j})<=max_edge_per_face && prod(double(ismember(F2{j}, idx_bounded)))==1
            if length(F2{j})<=max_edge_per_face && prod(double(ismember(indVert(F2{j}), idx_bounded)))==1
                Faces = [Faces; indVert(F2{j}) zeros(1,max_edge_per_face-length(F2{j}))];
            end
            indFace = indVert(tmp); % the indices of a face maps to original vertice index
            for l = 1:length(indFace)-1
                Edges = [Edges; indFace(l:l+1)];
            end
        end
    end
end

idxBoundedEdges = find(sum(ismember(Edges, idx_bounded),2)==2);
bounded_edges = Edges(idxBoundedEdges,:);

uniq_edges = unique(bounded_edges,'rows');

uniq_faces = unique(Faces, 'rows');