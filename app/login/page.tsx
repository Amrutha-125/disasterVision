"use client";

import { useForm } from "react-hook-form";
import Link from "next/link";
import { useRouter } from "next/navigation";

export default function LoginPage() {
  const { register, handleSubmit } = useForm();
  const router = useRouter();

  const onSubmit = (data: any) => {
    const user = localStorage.getItem("user");
    if (user) {
      const parsed = JSON.parse(user);
      if (
        parsed.email === data.email &&
        parsed.password === data.password
      ) {
        // Delay navigation slightly to allow alert to show first
        alert("Login successful!");
        setTimeout(() => {
          router.push("/prediction");
        }, 100); // 100ms delay
      } else {
        alert("Invalid credentials");
      }
    } else {
      alert("No user found. Please register first.");
    }
  };

  return (
    <div className="max-w-sm mx-auto bg-white p-6 rounded-xl shadow">
      <h2 className="text-2xl font-semibold mb-4">Login</h2>
      <form onSubmit={handleSubmit(onSubmit)} className="space-y-4">
        <input
          {...register("email")}
          type="email"
          placeholder="Email"
          className="w-full p-3 border rounded"
          required
        />
        <input
          {...register("password")}
          type="password"
          placeholder="Password"
          className="w-full p-3 border rounded"
          required
        />
        <button
          type="submit"
          className="w-full py-3 bg-blue-600 text-white rounded-xl"
        >
          Login
        </button>
      </form>
      <p className="text-center mt-4">
        Don&apos;t have an account?{" "}
        <Link href="/register" className="text-blue-600 underline">
          Register
        </Link>
      </p>
    </div>
  );
}
