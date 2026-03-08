import { vec3 } from "gl-matrix";
import type { Triangle } from "../uploadGlb";

export class BVHNode {
    private _minCorner: vec3;
    private _maxCorner: vec3;
    private _left: number;
    private _primitiveCount: number;

    constructor(minCorner: vec3, maxCorner: vec3) {
        this._minCorner = minCorner;
        this._maxCorner = maxCorner;
        this._left = -1;
        this._primitiveCount = -1;
    }

    set left(left: number) {
        this._left = left;
    }

    set primitiveCount(primitiveCount: number) {
        this._primitiveCount = primitiveCount;
    }

    set minCorner(minCorner: vec3) {
        this._minCorner = minCorner;
    }

    set maxCorner(maxCorner: vec3) {
        this._maxCorner = maxCorner;
    }

    get minCorner(): vec3 {
        return this._minCorner;
    }

    get maxCorner(): vec3 {
        return this._maxCorner;
    }

    get left(): number {
        return this._left;
    }

    get primitiveCount(): number {
        return this._primitiveCount;
    }
}

export class BVHTree {
    private _nodes: BVHNode[];
    private _nodesUsed: number;
    private _triangles: Triangle[];
    private _triangleIndices: number[];

    constructor(triangles: Triangle[]) {
        this._nodes = [];
        for (let i = 0; i < 2 * triangles.length - 1; i++) {
            this._nodes.push(new BVHNode(new Float32Array([Infinity, Infinity, Infinity]), new Float32Array([-Infinity, -Infinity, -Infinity])));
        }
        this._nodesUsed = 0;
        this._triangles = triangles;
        this._triangleIndices = [];
        for (let i = 0; i < triangles.length; i++) {
            this._triangleIndices.push(i);
        }

        this.build();
    }

    private build() {
        const rootNode: BVHNode = this._nodes[0];
        rootNode.left = 0;
        rootNode.primitiveCount = this._triangles.length;
        this._nodesUsed = 1;

        this._updateBounds(0);
        this._subdivide(0);
    }

    private _updateBounds(nodeIndex: number) {
        const node: BVHNode = this._nodes[nodeIndex];

        for (let i = 0; i < node.primitiveCount; i++) {
            const triangle: Triangle = this._triangles[this._triangleIndices[node.left + i]];

            for (const corner of triangle.positions) {
                vec3.min(node.minCorner, node.minCorner, corner);
                vec3.max(node.maxCorner, node.maxCorner, corner);
            }
        }
    }

    private _subdivide(nodeIndex: number) {
        const node: BVHNode = this._nodes[nodeIndex];

        if (node.primitiveCount <= 4) {
            return;
        }

        var extent: vec3 = [0.0, 0.0, 0.0];
        vec3.subtract(extent, node.maxCorner, node.minCorner);

        let axis: number = 0;
        if (extent[1] > extent[axis]) {
            axis = 1;
        }
        if (extent[2] > extent[axis]) {
            axis = 2;
        }

        const splitPosition: number = node.minCorner[axis] + extent[axis] * 0.5;

        let i: number = node.left;
        let j: number = node.left + node.primitiveCount - 1;

        while (i <= j) {
            if (this._triangles[this._triangleIndices[i]].centroid[axis] < splitPosition) {
                i++;
            } else {
                const temp: number = this._triangleIndices[i];
                this._triangleIndices[i] = this._triangleIndices[j];
                this._triangleIndices[j] = temp;
                j--;
            }
        }

        const leftCount: number = i - node.left;

        if (leftCount == 0 || leftCount == node.primitiveCount) {
            return;
        }

        const leftNodeIndex: number = this._nodesUsed;
        const rightNodeIndex: number = this._nodesUsed + 1;
        this._nodesUsed += 2;

        this._nodes[leftNodeIndex].left = node.left;
        this._nodes[leftNodeIndex].primitiveCount = leftCount;
        this._nodes[rightNodeIndex].left = i;
        this._nodes[rightNodeIndex].primitiveCount = node.primitiveCount - leftCount;

        node.left = leftNodeIndex;
        node.primitiveCount = 0;

        this._updateBounds(leftNodeIndex);
        this._updateBounds(rightNodeIndex);

        this._subdivide(leftNodeIndex);
        this._subdivide(rightNodeIndex);
    }

    get nodes(): BVHNode[] {
        return this._nodes;
    }

    get triangleIndices(): number[] {
        return this._triangleIndices;
    }

    get triangles(): Triangle[] {
        return this._triangles;
    }

    get nodesUsed(): number {
        return this._nodesUsed;
    }
}