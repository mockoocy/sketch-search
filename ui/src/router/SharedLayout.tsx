import { Toaster } from "@/general/components/sonner";
import { Outlet } from "@tanstack/react-router";

export function SharedLayout() {
  return (
    <div className="w-vw h-vh bg-gray-50 dark:bg-gray-900">
      <Outlet />
      <Toaster />
    </div>
  );
}
