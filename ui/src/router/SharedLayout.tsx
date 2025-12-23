import { Toaster } from "@/general/components/sonner";
import { Outlet } from "@tanstack/react-router";

export function SharedLayout() {
  return (
    <div className="min-h-screen flex items-center justify-center p-4">
      <Outlet />
      <Toaster />
    </div>
  );
}
