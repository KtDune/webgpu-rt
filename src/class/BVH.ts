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
    ...
}