"use client";

import { useForm } from "react-hook-form";
import Link from "next/link";

export default function RegisterPage() {
  const { register, handleSubmit } = useForm();

  const onSubmit = (data: any) => {
    localStorage.setItem("user", JSON.stringify(data));
    alert("Registered successfully! You can now log in.");
  };

  return (
    <div className="max-w-sm mx-auto bg-white p-6 rounded-xl shadow">
      <h2 className="text-2xl font-semibold mb-4">Register</h2>
      <form onSubmit={handleSubmit(onSubmit)} className="space-y-4">
        <input
          {...register("name")}
          placeholder="Name"
          className="w-full p-3 border rounded"
          required
        />
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
          Register
        </button>
      </form>
      <p className="text-center mt-4">
        Already have an account?{" "}
        <Link href="/login" className="text-blue-600 underline">
          Login
        </Link>
      </p>
    </div>
  );
}
