import type { DirectoryNode } from "@/gallery/schema";

export function findNodeByPath(
  node: DirectoryNode,
  path: string | undefined,
): DirectoryNode | null {
  if (path === undefined) return node;
  if (node.path === path) return node;

  for (const child of node.children) {
    const found = findNodeByPath(child, path);
    if (found) return found;
  }
  return null;
}
