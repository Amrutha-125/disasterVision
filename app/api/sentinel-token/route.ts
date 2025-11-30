// app/api/sentinel-token/route.ts

import { NextResponse } from "next/server";

export async function POST() {
  try {
    const res = await fetch("https://services.sentinel-hub.com/oauth/token", {
      method: "POST",
      headers: {
        "Content-Type": "application/x-www-form-urlencoded",
      },
      body: new URLSearchParams({
        client_id: process.env.NEXT_PUBLIC_SENTINEL_CLIENT_ID!,
        client_secret: process.env.NEXT_PUBLIC_SENTINEL_CLIENT_SECRET!,
        grant_type: "client_credentials",
      }),
    });

    if (!res.ok) {
      const errorText = await res.text();
      console.error("Sentinel token fetch failed:", errorText);
      return NextResponse.json({ error: "Token fetch failed" }, { status: 500 });
    }

    const { access_token } = await res.json();
    return NextResponse.json({ token: access_token });
  } catch (err) {
    console.error("Sentinel API error:", err);
    return NextResponse.json({ error: "Internal Server Error" }, { status: 500 });
  }
}
