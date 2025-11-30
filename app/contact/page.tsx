"use client";

import { useState } from "react";
import { useForm } from "react-hook-form";

export default function ContactPage() {
  const { register, handleSubmit, reset } = useForm();
  const [sent, setSent] = useState(false);

  const onSubmit = async (data: any) => {
    console.log(data);
    setSent(true);
    reset();
  };

  return (
    <div className="max-w-lg mx-auto">
      <h2 className="text-3xl font-semibold mb-6">Contact Us</h2>
      <form
        onSubmit={handleSubmit(onSubmit)}
        className="bg-white p-6 rounded-xl shadow"
      >
        <input
          {...register("name")}
          placeholder="Name"
          className="w-full p-3 border rounded mb-4"
          required
        />
        <input
          {...register("email")}
          placeholder="Email"
          type="email"
          className="w-full p-3 border rounded mb-4"
          required
        />
        <textarea
          {...register("message")}
          placeholder="Message"
          className="w-full p-3 border rounded mb-4"
          rows={5}
          required
        />
        <button
          type="submit"
          className="w-full py-3 bg-blue-600 text-white rounded-xl"
        >
          Send
        </button>
        {sent && (
          <p className="text-green-600 text-center mt-4">
            Thank you! We'll get back to you soon.
          </p>
        )}
      </form>
    </div>
  );
}
